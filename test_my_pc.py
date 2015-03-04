import os, sys

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
 
REVERSE_CAUSALITY = True

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
        G.add_node(id, level=child_level, order=i)
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
    # find all nodes without descendants
    assert nx.is_directed_acyclic_graph(G)
    nodes_stack = [node for node in G.nodes() 
                   if len(nx.ancestors(G, node)) == 0]
    expression_values = dict(
        (node, numpy.random.randn(n_timepoints)) 
        for node in nodes_stack )
    
    # deal with the descendants
    while len(nodes_stack) > 0:
        parent = nodes_stack.pop()
        for child in nx.descendants(G, parent):
            val = ( (1-corr)*numpy.random.randn(n_timepoints) 
                    + corr*expression_values[parent] )
            val = val/len(nx.ancestors(G, child))
            print val.sum(), len(nx.ancestors(G, child))
            print parent, child
            if child not in expression_values:
                expression_values[child] = val
            else:
                expression_values[child] += val
    
    return numpy.vstack(y for x, y in sorted(expression_values.items()))

def estimate_covariates(sample1, sample2, i):
    def cv_gen():
        n_tps = sample1.shape[1]
        yield range(n_tps), range(n_tps, 2*n_tps)
        yield range(n_tps, 2*n_tps), range(n_tps)
        return
    
    merged_samples = numpy.hstack((sample1, sample2))
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


def estimate_skeleton_from_samples(sample1, sample2, thresh=0.01):
    G = nx.DiGraph()
    for i in xrange(sample1.shape[0]):
        G.add_node(i)
        alpha, regression_coefs = estimate_covariates(sample1, sample2, i)
        print i, regression_coefs
        for j, val in enumerate(regression_coefs):
            if abs(val) >= thresh:
                G.add_edge(j, i, weight=val)
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
        neighbors = list(nx.all_neighbors(G, node))
        for n1 in neighbors:
            for n2 in neighbors:
                if n1 >= n2: continue
                if G.has_edge(n1, n2): continue
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
    #print a_fstat, b_fstat, critical_value
    if a_fstat < critical_value and b_fstat < critical_value:
        return True
    else:
        return False

def orient_v_structures(G, all_data):
    v_structures = set(iter_v_structures(G))
    for a, b, c in v_structures:
        if not are_cond_indep(a, b, c, all_data):
            # remove the edges point from c to a and b
            G.remove_edge(c, a)
            G.remove_edge(c, b)
            print "NOT", a, b, c
        else:
            pass
            print "%iT%i|%i" % (a, b, c)
    return

def apply_rule_1(b, c, G):
    """Check is there is a directed edge a->b such that a and b are not adjacent
    """
    for a in nx.ancestors(G, b):
        # skip bi-directed edges
        if G.has_edge(b, a): 
            continue
        # skip adjacent edges
        if G.has_edge(a, c) or G.has_edge(c, a):
            continue
        return True
    return False

def apply_rule_2(b, c, G):
    """Check is there is a directed edge a->b such that a and b are not adjacent
    """
    for a in nx.ancestors(G, b):
        # skip bi-directed edges
        if G.has_edge(b, a): 
            continue
        # skip adjacent edges
        if G.has_edge(a, c) or G.has_edge(c, a):
            continue
        return True
    return False

def main():
    real_G = simulate_causal_graph(1, 4)
    #nx.draw(real_G, nx.spring_layout(real_G,))
    #plt.show()
    #return

    sample1 = simulate_data_from_causal_graph(real_G, 100, 0.9)
    sample2 = simulate_data_from_causal_graph(real_G, 100, 0.9)
    all_data = numpy.hstack((sample1, sample2))
    est_G = estimate_skeleton_from_samples(sample1, sample2)
    orient_v_structures(est_G, all_data)
    while True:
        unoriented_edges = find_unoriented_edges(est_G)
        for a, b in unoriented_edges:
            if apply_rule_1(a, b, est_G):
                G.remove_edge(b ,a)
        break
    
    
    #nx.draw(est_G, nx.graphviz_layout(est_G,prog='twopi',args=''))
    nx.draw(est_G, nx.spring_layout(est_G,))
    plt.show()

    print partial_corr(all_data)
    return
    print 
    print partial_corr(numpy.hstack((sample1, sample2)))
    return

    inferred_digraph = nx.DiGraph(nodes=real_G.nodes())
    for parent, child in orient_v_structures(skeleton):
        print parent, child
        inferred_digraph.add_edge(parent, child)
    nx.draw(inferred_digraph)
    print nx.descendants(inferred_digraph, "0")
    print nx.descendants(real_G, "0")
    #nx.draw(real_G)
    plt.show()

main()
