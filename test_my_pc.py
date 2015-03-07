import os, sys

import math

from collections import defaultdict, OrderedDict

from itertools import combinations

import multiprocessing

import networkx as nx
import matplotlib.pyplot as plt

import random
import numpy.random
import numpy

from numpy.linalg import pinv

import numpy as np
from scipy import stats, linalg

from sklearn import linear_model, preprocessing
from sklearn.feature_selection import f_regression
from scipy.linalg import lstsq
from scipy.stats import f, norm, pearsonr

N_THREADS = 30

#print numpy.random.seed()
try: 
    numpy.random.seed(int(sys.argv[1]))
except:
    seed = random.randrange(100)
    print "SEED:", seed
    numpy.random.seed(seed)
# 82 is a good seed 

DEBUG_VERBOSE = False 
VERBOSE = True 
REVERSE_CAUSALITY = False

ALPHA = 0.01

def partial_corr(C):
    inv_cov = pinv(numpy.cov(C))
    normalization_mat = numpy.sqrt(
        numpy.outer(numpy.diag(inv_cov), numpy.diag(inv_cov)))
    return inv_cov/normalization_mat


def build_causal_graph(n_children, max_height):
    def add_children(G, parent, num):
        child_level = G.node[parent]['level'] + 1
        children_ids = []
        for i in xrange(num):
            id = "%i_%i_%i" % (child_level, G.node[parent]['order'], i)
            G.add_node(id, level=child_level, order=i, parent_order=G.node[parent]['order'])
            if not REVERSE_CAUSALITY:
                G.add_edge(parent, id)
            else:
                G.add_edge(id, parent)
            yield id
        return
    
    G = nx.DiGraph()
    G.add_node("0", level=0, order=0)
    pending_nodes = ["0",]
    while len(pending_nodes) > 0:
        curr_node = pending_nodes.pop()
        if G.node[curr_node]['level'] < max_height:
            pending_nodes.extend(list(add_children(G, curr_node, n_children)))
    return G

def simulate_causal_graph(depth, n_children):
    # build a causal graph
    return build_causal_graph(n_children, depth)


def simulate_data_from_causal_graph(G, n_timepoints, corr=0.9):
    # find all nodes without successors
    assert nx.is_directed_acyclic_graph(G)
    nodes_stack = [node for node in G.nodes() 
                   if len(G.predecessors(node)) == 0]
    expression_values = dict(
        (node, numpy.random.randn(n_timepoints)) 
        for node in nodes_stack )
    
    # deal with the successors
    while len(nodes_stack) > 0:
        parent = nodes_stack.pop()
        for child in G.successors(parent):
            val = ( (1-corr)*numpy.random.randn(n_timepoints) 
                    + corr*expression_values[parent] )
            val = val/len(G.predecessors(child))
            if child not in expression_values:
                expression_values[child] = val
            else:
                expression_values[child] += val
            nodes_stack.append(child)

    return ( sorted(expression_values), 
             numpy.vstack(y for x, y in sorted(expression_values.items())))

def estimate_covariates(sample1, sample2, resp_index, alpha=0.50):
    def cv_gen():
        n_tps = sample1.shape[1]
        yield range(n_tps), range(n_tps, 2*n_tps)
        yield range(n_tps, 2*n_tps), range(n_tps)
        return
    
    merged_samples = numpy.hstack((sample1, sample2))
    # normalize to sum 1
    merged_samples = ((merged_samples.T)/(merged_samples.sum(1))).T

    pred_mask = numpy.ones(merged_samples.shape[0], dtype=bool)
    pred_mask[resp_index] = False

    # filter marginally uncorrelated covariates
    response = merged_samples[resp_index,:]    
    best_i, min_pval = None, 10
    for i, row in enumerate(merged_samples):
        if i == resp_index: continue
        corr_coef, p_value = pearsonr(row, response)
        if p_value > alpha: pred_mask[i] = False
        if p_value < min_pval: best_i = i
    pred_mask[best_i] = True
    
    clf = linear_model.LassoCV(
       fit_intercept=True, cv=cv_gen(), verbose=0, max_iter=100000)
    clf.fit(merged_samples[pred_mask,:].T, response.T)
    
    alpha = clf.alpha_
    
    clf = linear_model.Lasso(
       alpha=alpha, max_iter=100000)
    clf.fit(merged_samples[pred_mask,:].T, response.T)

    coefs = clf.coef_.tolist()
    for i in sorted(int(x) for x in (1-pred_mask).nonzero()[0]):
        coefs.insert(i, 0)
    return alpha, coefs


def estimate_skeleton_from_samples(sample1, sample2, labels, thresh_ratio=1000):
    G = nx.DiGraph()
    for i in xrange(sample1.shape[0]):
        G.add_node(i, label=labels[i])
        alpha, regression_coefs = estimate_covariates(sample1, sample2, i)
        max_coef = max(numpy.abs(regression_coefs))
        for j, val in enumerate(regression_coefs):
            if abs(val) > 1e-6 and abs(val) >= max_coef/thresh_ratio:
                G.add_edge(i, j, weight=val)
    
    # remove non bi-directed edges and set the edge weights
    for a, b in list(G.edges()):
        if not G.has_edge(b, a): 
            G.remove_edge(a, b)
        else:
            G[a][b]['weight'] = min(G[a][b]['weight'], G[b][a]['weight'])
    
    return G.to_undirected()

def estimate_marginal_correlations(merged_samples, resp_index, 
                                   min_num_neighbors=1, 
                                   alpha=ALPHA):
    sig_samples = []
    response = merged_samples[resp_index,:]
    for i, row in enumerate(merged_samples):
        if i <= resp_index: continue
        corr_coef, p_value = pearsonr(row, response)
        if p_value < alpha or len(sig_samples) < min_num_neighbors: 
            sig_samples.append((p_value, corr_coef, i))
    sig_samples.sort(key=lambda x: (x[0], -x[1]))
    return sig_samples

def estimate_initial_skeleton_singlethread(normalized_data, labels, thresh_ratio=1000):
    # initialize a graph storing the covariance structure
    G = nx.Graph()
    for i in xrange(normalized_data.shape[0]):
        #if i >= 5: break
        marginal_dep = estimate_marginal_correlations(normalized_data, i)
        print i, normalized_data.shape[0], labels[i], len(marginal_dep)
        if not G.has_node(i): G.add_node(i, label=labels[i])
        for p, corr, j in marginal_dep:
            if not G.has_node(j): G.add_node(j, label=labels[j])
            G.add_edge(i, j, corr=corr, marginal_p=p)
    
    return G

def estimate_initial_skeleton(normalized_data, labels):
    # initialize a graph storing the covariance structure
    manager = multiprocessing.Manager()
    edges_and_data = manager.list()
    curr_node = multiprocessing.Value('i', 0)
    pids = []
    for p_index in xrange(N_THREADS):
        pid = os.fork()
        if pid == 0:
            while True:
                with curr_node.get_lock():
                    node = curr_node.value
                    if node >= normalized_data.shape[0]: break
                    curr_node.value += 1
                marginal_dep = estimate_marginal_correlations(
                    normalized_data,node)
                edges_and_data.append((node, marginal_dep ))
                print node, normalized_data.shape[0], len(marginal_dep)
            os._exit(0)
        else:
            pids.append(pid)
    try:
        for pid in pids:
            os.waitpid(pid, 0)
    except:
        for pid in pids:
            os.kill(pid, signal.SIGTERM)
        raise
    
    print "Building graph skeleton from marginal independence data"
    G = nx.Graph()
    for i, data in edges_and_data:
        if not G.has_node(i): G.add_node(i, label=labels[i])
        for p, corr, j in data:
            if not G.has_node(j): G.add_node(j, label=labels[j])
            G.add_edge(i, j, corr=corr, marginal_p=p)
    
    manager.shutdown()
    return G


def find_unoriented_edges(G):
    unoriented_edges = set()
    for start, stop in G.edges():
        if start > stop: continue
        if G.has_edge(stop, start):
            unoriented_edges.add((start, stop))
    return sorted(unoriented_edges)

def iter_unoriented_v_structures(G):
    for node, node_data in G.nodes(data=True):
        neighbors = [neighbor for neighbor in nx.all_neighbors(G, node)
                     if G.has_edge(neighbor, node) and G.has_edge(node, neighbor)]
        for n1 in neighbors:
            for n2 in neighbors:
                if n1 >= n2: 
                    continue
                if G.has_edge(n1, n2) or G.has_edge(n2, n1): 
                    continue
                yield n1, node, n2
    return

def iter_colliders(G):
    for node, node_data in G.nodes(data=True):
        predecessors = [predecessor for predecessor in G.predecessors(node)
                        if not G.has_edge(node, predecessor) 
                        and G.has_edge(predecessor, node)]
        for n1 in predecessors:
            assert G.has_edge(n1, node)
            assert not G.has_edge(node, n1)
            for n2 in predecessors:
                if n1 >= n2: continue
                yield n1, node, n2
    return

def orient_single_v_structure(G, data):
    for a, c, b in iter_unoriented_v_structures(G):
        if not are_cond_indep(a, b, c, data):
            # remove the edges point from c to a and b
            try: G.remove_edge(c, a)
            except: pass
            try: G.remove_edge(c, b)
            except: pass
            
            if VERBOSE:
                print "Orienting %s->%s<-%s" % (
                    G.node[a]['label'], G.node[c]['label'], G.node[b]['label'])
            return True
    
    return False

def orient_v_structures(G, ci_sets):
    for a, c, b in iter_unoriented_v_structures(G):
        if c not in ci_sets[(a,b)]:
            # remove the edges point from c to a and b
            try: G.remove_edge(c, a)
            except: pass
            try: G.remove_edge(c, b)
            except: pass
    return

def apply_rule_1(b, c, G):
    """Check is there is a directed edge a->b such that a and c are not adjacent
    """
    for a in G.predecessors(b):
        # skip bi-directed edges
        if G.has_edge(b, a): 
            continue
        # skip adjacent edges
        if G.has_edge(a, c) or G.has_edge(c, a):
            continue
        return True
    return False

def apply_rule_2(a, b, G):
    """Check is there is a directed edge a->b such that a and b are not adjacent
    """
    for c in G.successors(a):
        # skip bi-directed edges
        if G.has_edge(c, a): 
            continue
        # if there also exists a directed edge from c,b 
        # then rule 2 applies
        if G.has_edge(c, b) and not G.has_edge(b, c):
            return True
    return False

def apply_rule_3(a, b, G):
    """Check is there are two chains a--c->b and a--d->b such that 
       c and d are not adjacent.
    """
    intermediate_nodes = set()
    # find all such chains
    for c in G.successors(a):
        # skip bi-directed edges
        if G.has_edge(c, a) and G.has_edge(c, b) and not G.has_edge(b, c): 
            intermediate_nodes.add(c)

    # try to find a pair of non adjacent nodes
    for c in intermediate_nodes:
        for d in intermediate_nodes:
            if c == d: continue
            if not G.has_edge(c, d) and not G.has_edge(d, c):
                return True
    return False

def apply_rule_4(a, b, G):
    """Check is there is a chain a--c->d->b where c and b are non adjacent.
    """
    # find all such chains
    for c in G.successors(a):
        # skip non bi-directed edges
        if not G.has_edge(c, a):
            continue
        # skip c that are adjacent to b
        if G.has_edge(c, b) or G.has_edge(b, c):
            continue
        for d in G.successors(c):
            # skip bi directed edges
            if G.has_edge(d, c): continue
            if G.has_edge(d, b) and not G.has_edge(b, d):
                return True
    return False

def apply_IC_rules(G):
    unoriented_edges = find_unoriented_edges(G)
    for a, b in unoriented_edges:
        if apply_rule_1(a, b, G):
            if DEBUG_VERBOSE: print "Applying Rule 1:", a, b
            G.remove_edge(b ,a)
            return True
        elif apply_rule_2(a, b, G):
            if DEBUG_VERBOSE: print "Applying Rule 2:", a, b
            G.remove_edge(b ,a)
            return True
        elif apply_rule_3(a, b, G):
            if DEBUG_VERBOSE: print "Applying Rule 3:", a, b
            G.remove_edge(b ,a)
            return True
        elif apply_rule_4(a, b, G):
            if DEBUG_VERBOSE: print "Applying Rule 4:", a, b
            G.remove_edge(b ,a)
            return True
    return False


def break_cycles(G):
    G = G.copy()
    cycles = nx.cycle_basis(G)
    while len(cycles) > 0:
        # break cycles
        edge_cnts = defaultdict(int)
        edge_weights = {}
        for cycle in nx.cycle_basis(G):
            for a,b in zip(cycle[:-1], cycle[1:]) + [(cycle[-1],cycle[0]),]:
                a, b = min(a,b), max(a,b)
                edge_cnts[(a,b)] += 1
                edge_weights[(a,b)] = G[a][b]['weight']
        edge = sorted(
            edge_cnts, key=lambda x: (-edge_cnts[x], abs(edge_weights[x])))[0]
        #print edge, edge_cnts[edge], abs(edge_weights[edge])
        #print "Removing edge:", edge
        G.remove_edge(*edge)
        cycles = nx.cycle_basis(G)
    return G

def test_for_CI(G, n1, n2, normalized_data, order, alpha):
    """Test if n1 and n2 are conditionally independent. 

    If they are not return None, else return the conditional independence set.
    """
    N = normalized_data.shape[1]
    ones = numpy.ones((N,1), dtype=float)
    
    n1_resp = normalized_data[n1,:]    
    n1_neighbors = set(G.neighbors(n1))    
    
    n2_resp = normalized_data[n2,:]
    n2_neighbors = set(G.neighbors(n2))
    
    common_neighbors = n1_neighbors.intersection(n2_neighbors) - set((n1, n2))
    # if there aren't enough neighbors common to n1 and n2, return none
    if len(common_neighbors) < order: 
        return None
    
    min_score = 1e100
    best_p_val = None
    best_neighbors = None
    n_common_neighbors = 0
    for covariates in combinations(common_neighbors, order):
        n_common_neighbors += 1
        predictors = numpy.hstack(
            (ones, normalized_data[numpy.array(covariates),:].T))
        # test if node is independent of neighbors given for some subset
        rv1, _, _, _ = lstsq(predictors, n1_resp)
        rv2, _, _, _ = lstsq(predictors, n2_resp)
        cor, pval =  pearsonr(n1_resp - rv1.dot(predictors.T), 
                              n2_resp - rv2.dot(predictors.T))
        if abs(cor) < min_score:
            min_score = abs(cor)
            best_neighbors = covariates
            best_p_val = pval
        #score = math.sqrt(N-order-3)*0.5*math.log((1+cor)/(1-cor))
        #print abs(score),  norm.isf(alpha/(len(neighbors)*2)), cor, pval
        #if abs(score) < norm.isf(alpha/(len(neighbors)*2)): 
    
    # make the multiple testing correction
    if best_p_val/n_common_neighbors < alpha:
        return None
    else:
        return best_neighbors

def apply_pc_iteration(G, normalized_data, order, alpha=ALPHA):    
    cond_independence_sets = defaultdict(set)
    
    ## we can't estimate higher order interactiosn than we have samples
    #N = normalized_data.shape[1]
    #if N - order - 3 <= 0:
    #    return cond_independence_sets

    num_nodes = len(G.nodes())
    for n1 in G.nodes():    
        n1_neighbors = sorted(
            G.neighbors(n1), key=lambda n2: G[n1][n2]['marginal_p'])
        num_neighbors = len(n1_neighbors)
        if VERBOSE:
            print "Test O%i: %i/%i %i neighbors ... " % (
                order, n1, num_nodes, num_neighbors),

        for n2 in n1_neighbors:
            if n2 <= n1: continue
            are_CI = test_for_CI(G, n1, n2, normalized_data, order, alpha)
            if are_CI == None:
                pass
                if DEBUG_VERBOSE: print "%i NOT CI of %i" % (n1, n2)
            else:
                if DEBUG_VERBOSE:
                    print "%i IS CI %i | %s" % (
                        n1, n2, ",".join(str(x) for x in are_CI))
                cond_independence_sets[(n1, n2)].add(are_CI)
                G.remove_edge(n1, n2)
                num_neighbors -= 1
        if VERBOSE: 
            print "%i remain (%i removed)" % (
                num_neighbors, len(n1_neighbors) - num_neighbors)
    
    return cond_independence_sets

def estimate_pdag(sample1, sample2, labels):
    # combine and normalize the samples
    normalized_data = numpy.hstack((sample1, sample2))
    normalized_data = ((normalized_data.T)/(normalized_data.sum(1))).T

    skeleton = estimate_initial_skeleton(normalized_data, labels)
    nx.write_gml(skeleton, "skeleton.gml")
    
    curr_pdag = skeleton.copy()
    cond_independence_sets = defaultdict(set)
    for i in xrange(1,normalized_data.shape[1]-3+1):
        inner_cis = apply_pc_iteration( curr_pdag, normalized_data, i )
        for key, vals in inner_cis.items():
            cond_independence_sets[key].update(vals)
        print i, len(nx.connected_component_subgraphs(curr_pdag))
    
    #skeleton = break_cycles(skeleton)
    est_G = skeleton.to_directed()
    orient_v_structures(est_G, cond_independence_sets)
    applied_rule = True
    while apply_IC_rules(est_G): pass
    #assert False
    return est_G, cond_independence_sets

def hierarchical_layout(real_G):
    level_grouped_nodes = defaultdict(list)
    for node, data in real_G.nodes(data=True):
        level_grouped_nodes[data['level']].append(node)
    pos = {}
    for level, nodes in level_grouped_nodes.iteritems():
        y_pos = 1 - float(level+1)/(len(level_grouped_nodes)+1)
        for i, node in enumerate(sorted(
                nodes, key=lambda id: [int(x) for x in id.split("_")])):
            x_pos = float(i+1)/(len(nodes)+1)
            pos[node] = numpy.array((x_pos, y_pos+(x_pos/1.7)**2))
    
    return pos

def iter_all_subsets_of_siblings(G, node):
    siblings = set(nx.all_neighbors(G, node))
    for i in xrange(1, len(siblings)+1):
        for subset in combinations(siblings, i):
            yield subset
    return

def brute_force_find_all_consistent_dags(pdag, data):
    def get_undirected_edges(G):
        undirected_edges = []
        for a, b in G.edges():
            if G.has_edge(b, a):
                undirected_edges.append( (a,b) )
        return undirected_edges
    
    def orient_edge_and_propogate_changes(G, a, b):
        G = G.copy()
        G.remove_edge(b, a)
        #orient_v_structures(G, data)
        while apply_IC_rules(G): pass
        return G

    # the dags that we've found
    dags = []
    # stack containing both edges that we have already oriented, and 
    # the current pdag after those edges have been oriented
    stack = [ [[(a,b),], orient_edge_and_propogate_changes(pdag, a, b)]
              for a, b in get_undirected_edges(pdag) ]
    while len(stack) > 0:
        oriented_edges, curr_pdag = stack.pop()
        edges_to_orient = get_undirected_edges(curr_pdag)
        # if there are no more edges to orient, then make sure that we haven't 
        # already seen it and that it is acyclic, and add it
        if len(edges_to_orient) == 0: 
            if (not any(set(curr_pdag.edges()) == set(x.edges()) for x in dags)
                    and nx.is_directed_acyclic_graph(curr_pdag)):
                dags.append(curr_pdag)
        else:
            for a, b in edges_to_orient:
                new_pdag = orient_edge_and_propogate_changes(curr_pdag, a, b)
                stack.append([oriented_edges + [(a, b),], new_pdag])
    
    return dags


def find_all_consistent_dags(pdag, cond_independence_sets, real_G):
    initial_colliders = set(iter_colliders(pdag))
    
    def get_undirected_edges(G):
        undirected_edges = []
        for a, b in G.edges():
            if G.has_edge(b, a):
                undirected_edges.append( (a,b) )
        return undirected_edges
    
    def orient_edge_and_propogate_changes(G, a, b):
        G = G.copy()
        G.remove_edge(b, a)
        orient_v_structures(G, cond_independence_sets)
        while apply_IC_rules(G): pass
        # make sure that the resulting graph is acyclic
        #if not nx.is_directed_acyclic_graph(G): return None
        if any( collider not in initial_colliders 
                for collider in  iter_colliders(G) ): 
            return None
        return G

    #unoriented_v_structures = list(iter_unoriented_v_structures(pdag))
    # the dags that we've found
    dags = []
    # stack containing both edges that we have already oriented, and 
    # the current pdag after those edges have been oriented
    stack = [ [[(a,b),], orient_edge_and_propogate_changes(pdag, a, b)]
              for a, b in get_undirected_edges(pdag) ]
    stack = [(x, g) for x, g in stack if g != None]
    while len(stack) > 0:
        oriented_edges, curr_pdag = stack.pop()
        edges_to_orient = get_undirected_edges(curr_pdag)
        # if there are no more edges to orient, then make sure that we haven't 
        # already seen it and that it is acyclic, and add it
        if len(edges_to_orient) == 0: 
            if (not any(set(curr_pdag.edges()) == set(x.edges()) for x in dags)
                    and nx.is_directed_acyclic_graph(curr_pdag)):
                #plot_pdag(curr_pdag, real_G)
                dags.append(curr_pdag)
        else:
            for a, b in edges_to_orient:
                new_pdag = orient_edge_and_propogate_changes(curr_pdag, a, b)
                if new_pdag != None:
                    stack.append([oriented_edges + [(a, b),], new_pdag])
    
    return dags


def plot_pdag(pdag, real_G):
    real_G_layout = hierarchical_layout(real_G)
    #nx.draw(est_G, nx.graphviz_layout(est_G,prog='twopi',args=''))
    labels = dict((id, data['label']) for id, data in pdag.nodes(data=True))
    pos = dict((id, real_G_layout[data['label']]) 
               for id, data in pdag.nodes(data=True))
    nx.draw(pdag, pos, labels=labels, node_size=1500, node_color='white')
    plt.show()
    return

def test():
    real_G = simulate_causal_graph(2, 2)
    #nx.draw(real_G, hierarchical_layout(real_G), node_size=1500, node_color='blue')
    #plt.show()
    #return

    labels, sample1 = simulate_data_from_causal_graph(real_G, 3, 0.5)
    labels, sample2 = simulate_data_from_causal_graph(real_G, 3, 0.5)
    #print partial_corr(preprocessing.scale(numpy.hstack((sample1, sample2)).T).T)
    #return
    pdag, cond_independence_sets = estimate_pdag(sample1, sample2, labels)
    plot_pdag(pdag, real_G)
    for x in find_all_consistent_dags(
            pdag, cond_independence_sets, real_G):
        print x.edges()
        plot_pdag(x, real_G)
    
    

    return

def load_data():
    samples = None
    genes = []
    expression = []
    with open("all_quant.txt") as fp:
        for i, line in enumerate(fp):
            data = line.split()
            if i == 0:
               samples = data[1:]
               continue
            # append the gene name
            gene_expression = numpy.array(map(float, data[1:]))
            if numpy.max(gene_expression) < 100: continue
            genes.append(data[0])
            # add random normal noise to the gene expressionvalues to prevent 
            # high correlation artifacts due to rounding error, etc. 
            gene_expression += (numpy.random.randn(len(gene_expression)))**2
            expression.append( gene_expression )
    
    expression = numpy.vstack(expression)
    s1 = expression[:,(3,0,5)]
    s2 = expression[:,(4,2,7)]
    return genes, s1, s2

def main():
    genes, sample1, sample2 = load_data()
    pdag, cond_independence_sets = estimate_pdag(sample1, sample2, genes)
    nx.write_gml(pdag.to_undirected(pdag), "expression_GT_3000_pdag.gml")
    print pdag
    return

main()
