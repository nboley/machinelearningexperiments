import os, sys

from collections import defaultdict

from itertools import combinations

import networkx as nx
import matplotlib.pyplot as plt

import random
import numpy.random
import numpy

from numpy.linalg import pinv

import numpy as np
from scipy import stats, linalg

from sklearn import linear_model
from sklearn.feature_selection import f_regression
from scipy.linalg import lstsq
from scipy.stats import f

#print numpy.random.seed()
#numpy.random.seed(int(sys.argv[1]))

VERBOSE = False 
REVERSE_CAUSALITY = False

def partial_corr(C):
    inv_cov = pinv(numpy.cov(C))
    normalization_mat = numpy.sqrt(
        numpy.outer(numpy.diag(inv_cov), numpy.diag(inv_cov)))
    return inv_cov/normalization_mat

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

def build_causal_graph(n_children, max_height):
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

def estimate_covariates(sample1, sample2, i):
    def cv_gen():
        n_tps = sample1.shape[1]
        yield range(n_tps), range(n_tps, 2*n_tps)
        yield range(n_tps, 2*n_tps), range(n_tps)
        return
    
    merged_samples = numpy.hstack((sample1, sample2))
    # normalize to sum 1
    merged_samples = (merged_samples.T/(merged_samples.sum(1)+1)).T
    #print merged_samples
    #assert False
    mask = numpy.ones(merged_samples.shape[0], dtype=bool)
    mask[i] = False
    
    response = merged_samples[i,:]
    
    clf = linear_model.LassoCV(
       fit_intercept=True, cv=cv_gen(), verbose=0, max_iter=100000)
    clf.fit(merged_samples[mask,:].T, response.T)
    
    alpha = clf.alpha_
    coefs = clf.coef_.tolist()
    coefs.insert(i, 0)
    
    return alpha, coefs


def estimate_skeleton_from_samples(sample1, sample2, labels, thresh_ratio=10):
    G = nx.Graph()
    for i in xrange(sample1.shape[0]):
        G.add_node(i, label=labels[i])
        alpha, regression_coefs = estimate_covariates(sample1, sample2, i)
        max_coef = max(regression_coefs)
        print i, alpha, regression_coefs
        for j, val in enumerate(regression_coefs):
            if abs(val) >= max_coef/thresh_ratio:
                G.add_edge(i, j, weight=val)
    return G

def find_unoriented_edges(G):
    unoriented_edges = set()
    for start, stop in G.edges():
        if start > stop: continue
        if G.has_edge(stop, start):
            unoriented_edges.add((start, stop))
    return sorted(unoriented_edges)

def iter_v_structures(G):
    for node, node_data in G.nodes(data=True):
        neighbors = [neighbor for neighbor in nx.all_neighbors(G, node)
                     if G.has_edge(neighbor, node) and G.has_edge(node, neighbor)]
        for n1 in neighbors:
            for n2 in neighbors:
                if n1 >= n2: 
                    continue
                if G.has_edge(n1, n2) or G.has_edge(n2, n1): 
                    continue
                yield n1, n2, node
    return

def are_cond_indep(a, b, c, data):
    N = data.shape[1]
    ones = numpy.ones((N,1), dtype=float)
    a_val = data[(a,),:].T
    b_val = data[(b,),:].T
    c_val = data[(c,),:].T

    sln, rss_1, rank, s = lstsq(numpy.hstack((ones, c_val)), a_val)
    sln, rss_2, rank, s = lstsq(numpy.hstack((ones, c_val, b_val)), a_val)
    a_fstat = N*(rss_1 - rss_2)/rss_2

    sln, rss_1, rank, s = lstsq(numpy.hstack((ones, c_val)), a_val)
    sln, rss_2, rank, s = lstsq(numpy.hstack((ones, c_val, b_val)), a_val)
    b_fstat = N*(rss_1 - rss_2)/rss_2

    critical_value = f.isf([1e-2,], 1, N-2)
    #print a, b, c, a_fstat, b_fstat, critical_value
    if a_fstat < critical_value and b_fstat < critical_value:
        return True
    else:
        return False

def orient_single_v_structure(G, data):
    for a, b, c in iter_v_structures(G):
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
        else:
            continue
            ### THIS IS WRONG 
            # remove the edges point from c to a and b
            try: G.remove_edge(a, c)
            except: pass
            try: G.remove_edge(b, c)
            except: pass
            
            if VERBOSE:
                print "Orienting %s->%s<-%s" % (
                    G.node[a]['label'], G.node[c]['label'], G.node[b]['label'])
            return True
    return False

def orient_v_structures(G, data):
    while orient_single_v_structure(G, data): pass
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
            if VERBOSE: print "Applying Rule 1:", a, b
            G.remove_edge(b ,a)
            return True
        elif apply_rule_2(a, b, G):
            if VERBOSE: print "Applying Rule 2:", a, b
            G.remove_edge(b ,a)
            return True
        elif apply_rule_3(a, b, G):
            if VERBOSE: print "Applying Rule 3:", a, b
            G.remove_edge(b ,a)
            return True
        elif apply_rule_4(a, b, G):
            if VERBOSE: print "Applying Rule 4:", a, b
            G.remove_edge(b ,a)
            return True
    return False

def estimate_pdag(sample1, sample2, labels):
    skeleton = estimate_skeleton_from_samples(sample1, sample2, labels)
    est_G = skeleton.to_directed()
    orient_v_structures(est_G, numpy.hstack((sample1, sample2)))
    applied_rule = True
    while apply_IC_rules(est_G): pass
    return est_G

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
            pos[node] = numpy.array((x_pos, y_pos))
    
    return pos

def iter_all_subsets_of_siblings(G, node):
    siblings = set(nx.all_neighbors(G, node))
    for i in xrange(1, len(siblings)+1):
        for subset in combinations(siblings, i):
            yield subset
    return

def plot_pdag(pdag, real_G):
    real_G_layout = hierarchical_layout(real_G)
    #nx.draw(est_G, nx.graphviz_layout(est_G,prog='twopi',args=''))
    labels = dict((id, data['label']) for id, data in pdag.nodes(data=True))
    pos = dict((id, real_G_layout[data['label']]) 
               for id, data in pdag.nodes(data=True))
    nx.draw(pdag, pos, labels=labels, node_size=1500, node_color='white')
    plt.show()
    return

def main():
    real_G = simulate_causal_graph(1, 2)
    #nx.draw(real_G, layout, node_size=1500, node_color='blue')
    #plt.show()
    #return

    labels, sample1 = simulate_data_from_causal_graph(real_G, 100, 0.5)
    labels, sample2 = simulate_data_from_causal_graph(real_G, 100, 0.5)
    
    print "Penalized Regression:"
    pdag = estimate_pdag(sample1, sample2, labels)
    
    for x in iter_all_subsets_of_siblings(pdag, 0):
        print x

    plot_pdag(pdag, real_G)

    return

main()
