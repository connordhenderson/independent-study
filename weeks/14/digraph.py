#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.debugger import set_trace
from collections import Counter
from functools import reduce 
import operator
import itertools
from scipy.stats import median_absolute_deviation

# util
import os
import re
import random
import numpy as np

# visualization & graphing;
import graphviz
import networkx as nx
from matplotlib import pyplot as plt
from graphviz import render

get_ipython().run_line_magic('matplotlib', 'inline')

fdir = "../data/queries_explains_10g/queries1tb/"


# ## Plan Details

# In[2]:


PLAN_COSTS = ['Cumulative Total Cost',
'Cumulative CPU Cost',
'Cumulative I/O Cost',
'Cumulative Re-Total Cost',
'Cumulative Re-CPU Cost',
'Cumulative Re-I/O Cost',
'Cumulative First Row Cost']


# ## Regex

# In[3]:


access_regex = re.compile(r"(Access Plan:)+")
plan_details_regex = re.compile(r"(Plan Details:)+")

# get the relationships for the parent/child nodes
relationship_regex = re.compile(r"(\/|\+)(-+\+)+-+(\\|\+)|\|")    #5

# get the node type or table schema
type_regex = re.compile(r"(\w+((:(\+|-| )*)|-)*)+\w+")            #1

# get the node label or table name
label_regex = re.compile(r"\d+|\w+")                              #2

# separates by WS
not_ws_regex = re.compile(r"\S+")                                 #3

# looks for various float representations
float_regex = re.compile(r"(\d+(.\d+){0,1}(e(\+|-)\d+){0,1})")    #

# used to find the labels for plan details
pd_label_regex = re.compile(r"^\s?\d+\)")

# strips '#-#-' prefix from the label of form '#-#-@' by ignoring the #-#- capture group
strip_label_regex = re.compile(r"(?:(\w+-){2})(\S+)")


# ## Visualization

# In[4]:


def visualize_graph(g, view=False, root=None):
    if root is None:
        nodes = g.nodes
        edges = g.edges
    else:
        nodes, edges = g.get_graph()[root].get_subgraph()
    
    _g = graphviz.Digraph('g', node_attr={'shape': 'record', 'height': '.1'})
    for n in nodes:
        if str(n[0]) == "0":
            continue
        _g.node(str(n[0]), str(n[1]))

    for e in edges:
        if str(e[1]) == "0":
            continue
        _g.edge(str(e[0]), str(e[1]))

    if view:
        try:
            _g.view()
        except:
            pass
    else:
        return _g
    
    
def visualize_joins(g):
    joins = g.get_graph()['1-0-1'].get_join_nodes()
    joins = [visualize_graph(g, root=j) for j in joins]
    return joins


# ## Helper Functions

# In[5]:


def save_dict(f, dct):
    np.save('../data/output_qep/' + f + '.npy', dct) 

def load_dict(f):
    # try to load a saved dict
    result = {}
    try:
        result = np.load('../data/output_qep/' + f + '.npy',allow_pickle='TRUE').item()
    except FileNotFoundError:
        print('no ' + f + ' file found at \'../data/output_qep/' + f + '.npy\'')
    return result

def canonical(s):
    return "temp" if s.isdigit() else s

# returns the center (float) point between two ints in a tuple
def center(tup):
    return (tup[0] + tup[1])/2

# determines the size of the span of a tuple of int's
def span(s):
    return s[1] - s[0]


"""
@param  rels   list(tup):  a list of tuples that represent the spans of the given edges for a line
@param  val    int:        the value of the center which we'd like to find the containing edge for
@return result int:        the index of the relationship that contains the edge
"""
def get_parent_relationship(rels, val):
    # added dummy -2-1 so we can return -1 for no match
    result = np.argmax([val > rel[0] and val <= rel[1] for rel in [(-2,-1)] + rels], -1)
    return result - 1

def strip_label(label):
    return strip_label_regex.search(label).group(2)


def test_ownership(r, c, allow_doubles = False):
    result = [get_parent_relationship(r, _c) for _c in c]
    
    if allow_doubles:
        return result
    
    for i in range(1, len(result)):
        # if we encounter edges sequentially and they aren't -1 then find the closest center
        if result[i - 1] == result[i] and result[i] > 0:
            if abs(c[i] - center(r[result[i]])) >= abs(c[i - 1] - center(r[result[i]])):
                result[i] = -1
            else:
                result[i - 1] = -1
    return result


# ## Nodes

# In[6]:


# TODO:
#      -> deal with IDs consistently (str, int, label?)

"""
The nodes of the graph represent the low level plan operators (LOLEPOPS) in the query execution plan;
ultimately we'd like to expand on Node2Vec to include additional node features
@param idx       (int):     node index
@param parent    (node):    parent node
@param label     (int):     node class label from node_types dict
@param attr      (list):    additional nodes/features (NYI)
"""
class node:
    def __init__(self, idx, attr=None):        
        self.idx   = idx
        self.attr  = attr
        self.children = []
        
    def insert(self, node):
        self.children.append(node)
    
    def get_children(self):
        return self.children
        
    def __str__(self):
        return f"[id#{self.idx}: children: {[c.idx for c in self.get_children()]}]"
    
    def get_all_terminal(self):
        if self.children == []:
            return [self.idx]
        result =  reduce(operator.concat, [child.get_all_terminal() for child in self.get_children()])
        return result
    
    def get_join_nodes(self, current_node = None):
        if current_node is None:
            current_node = self
            
        joins = []
        children = current_node.get_children()
        if len(children) > 1:
            joins.append(current_node.idx)
        elif children == []:
            return []

        for child in children:
            joins += child.get_join_nodes(child)
        return joins
    
    def get_subgraph(self, parent = None):
        edges = []
        nodes = []
        
        if parent is not None:
            edges.append((strip_label(parent.idx), strip_label(self.idx)))
            nodes.append((strip_label(self.idx),strip_label(self.idx)))
            
        if len(self.children) > 0:
            for child in self.children:
                
                child_nodes, child_edges = child.get_subgraph(self)
                
                for edge in child_edges:
                    edges.append(edge)
                    
                for node in child_nodes:
                    nodes.append(node)
                    
        return nodes, edges 


# ## Graph Parsing

# In[7]:


"""
Create a directed graph-structure that stores a list of nodes and edges (pairs of node indices) of 
a graph from a SQL Explain Plan. The structure can be parsed from the 'Action Plan' and the features can
be extracted from the 'Plan Details'
@param exfmt (str):  SQL Explain Plan from DB2
"""
class digraph:
    def __init__(self, exfmt):
        self.file = fdir + exfmt
        self.head = None
        self.node_dict = {}
        
        self.nodes, self.edges, self.labels = self.get_access_plan()
        self.plan_details = self.get_plan_details()
        
    """
    Get Access Plan
    """
    def get_access_plan(self):
        node_types = load_dict('node_types')
        node_labels = load_dict('node_labels')
        
        ap_start = None
        ap_end   = None
        
        prv_nodes = None
        edges = None
        prv_edges = None
        
        _nodes = []
        _edges = []
        _labels = {}
        
        # tracking line-counts for debugging & printing portions of the explain plan
        lx       = 0
        lines    = ''
        depth    = 0
        
        with open(self.file) as f:
            lines = f.readlines()
            
        # getting the access plan:    
        # iteration over sql explain plan format file
        for line in lines:
            # incr. line index
            lx += 1

            # track access plan boundaries for debugging
            if ap_start is None:
                if access_regex.search(line):
                    ap_start = lx + 5 #offset
            elif ap_end is None and line[0] == '-' and lx - ap_start > 1:
                ap_end = lx

        lines = lines[ap_start:ap_end]
        
        lx       = ap_start
        # parsing the AP 6 lines at a time (height of the layers in the explain plan):
        while len(lines) >= 6:        
            lx += 6
            depth += 1
            
            # create the nodes
            # get table-names; figure out where they're centered and identify relationship. first, remove ws after ':'
            node_cardinality = [l for l in not_ws_regex.finditer(lines[0])]
            
            # convert node types to IDs so they're consistent between query plans. if it's not in a dict, put it in
            node_type = [l for l in type_regex.finditer(lines[1])]
            for i in range(len(node_type)):
                try:
                    node_types[str(node_type[i].group(0))]
                except KeyError:
                    node_types[str(node_type[i].group(0))] = len(node_types) + 1
                    continue  

            # convert node labels to IDs so they're consistent between query plans. if it's not in a dict, put it in
            node_label = [l for l in label_regex.finditer(lines[2])]
            for i in range(len(node_label)):
                try:
                    node_labels[str(node_label[i].group(0))]
                except KeyError:
                    node_labels[str(node_label[i].group(0))] = len(node_labels) + 1
                    continue          
            
            node_attr = [l for l in not_ws_regex.finditer(lines[3])]
            
            # put them into an easy to work with vector form
            nodes = np.vstack((node_cardinality, node_type, node_label, node_attr)).T           
            
            # get the indices of the node features with the longest length
            centers = [np.argmax([len(n_.group(0)) for n_ in n]) for n in nodes]
            # get the matches those of indices; find the center of the spans of those matches
            centers = [int(center(nodes[i][centers[i]].span())) for i in range(len(nodes))]
            
            # get edges; expand boundaries by 2 to account for 2 char buffer on each side
            edges = [(r.span()[0] - 2, r.span()[1] + 2) for r in relationship_regex.finditer(lines[5])] 

            edge_owner = test_ownership(edges, centers)
            
            # add the nodes to the node list
            for i in range(len(nodes)):
                _nodes.append((str(depth)+'-'+str(i)+'-'+nodes[i][2].group(0), nodes[i][2].group(0)))
                _labels[nodes[i][2].group(0)] = nodes[i][1].group(0)
    
            if prv_edges == None:
                # continue to the next layer
                self.head = nodes[0]
                pass

            else:
                # BUG: some overlapping tables are getting edges when they shouldn't be (check empty (4) spot?)
                # get all possible edges that could be the parents; grab the closest match and use it's index
                candidate_parents = test_ownership(prv_edges, centers, True)
                #print(candidate_parents)
                for i in range(len(candidate_parents)):
                    for j in range(len(prv_edge_owner)):
                        if candidate_parents[i] == prv_edge_owner[j]:
                            #e = (prv_nodes[j][2].group(0) , nodes[i][2],group(0))
                            
                            # "depth-node#-label"
                            e1 = str(depth - 1)+'-'+str(j)+'-'+prv_nodes[j][2].group(0)
                            e2 = str(depth)+'-'+str(i)+'-'+ nodes[i][2].group(0)
                            
                            _edges.append((e1,e2))
                            #print(e)

            # continue to the next layer
            prv_nodes      = nodes
            prv_edges      = edges
            prv_centers    = centers
            prv_edge_owner = edge_owner
            lines = lines[6:]

        save_dict('node_types', node_types)
        save_dict('node_labels', node_labels)
        return _nodes, _edges, _labels

    """
    Plan Details
    """
    def get_plan_details(self):
        pd_start = None
        pd_end   = None
        lx       = 0
        lines    = ''
        
        plan_details = {}
        
        with open(self.file) as f:
            lines = f.readlines()
        
        for line in lines:
            lx += 1
        
            if pd_start is None:
                if plan_details_regex.search(line):
                    pd_start = lx + 1 #offset
            elif pd_end is None and line[0] == '-' and lx - pd_start > 1:
                pd_end = lx
        lines = lines[pd_start:pd_end]

        while len(lines) > 9:
            # get the node label '\S?#)' and extract label
            label = pd_label_regex.search(lines[0])
            if label:
                label = float_regex.search(label.group(0)).group(0)
                
                # get the costs and associate them to the node
                node_costs = [float_regex.search(lines[i + 1]).group(0) for i in range(len(PLAN_COSTS))]
                plan_details[int(label)] = node_costs
                
            lines = lines[1:]
        return plan_details
    
    # Get the joins that are present in the table using the local labels
    def get_joins(self, node_types = None, terminal_dict = None):
        if node_types is None:
            node_types = load_dict('node_types')
        if terminal_dict is None:
            terminal_dict = load_dict('terminal_dict')
            
        nodes = self.get_graph()

        # Get the nodes that represent a join in the graph (by counting # of children)
        joins = Counter([edge[0] for edge in self.edges])
        joins = [j for j in joins if joins[j] >= 2]

        # get all the terminal nodes from those joins
        joins = [(j, nodes[j].get_all_terminal()) for j in nodes if j in joins]
        
        # strip the labels
        joins = np.array([(strip_label(j[0]), [strip_label(_j) for _j in j[1]]) for j in joins], dtype=object)
     
        return joins
    
    # return a list of all of the terminal nodes (tables? the ordinal values returned are temp. tables)
    def get_terminal_nodes(self):
        res = np.setdiff1d([e[1] for e in g.edges], [e[0] for e in g.edges]).tolist()
        res = np.array([strip_label(e) for e in res])
        return res
    
    def get_graph(self):
        nodes = {}
        for edge in self.edges:
            if edge[0] not in nodes:
                nodes[edge[0]] = node(edge[0])
            if edge[1] not in nodes:
                nodes[edge[1]] = node(edge[1])
            nodes[edge[0]].insert(nodes[edge[1]])
        return nodes
    
    def qep2vec(self, degree = 0):
        joins = self.get_joins()
            
        if degree > 0:
            joins = np.array([join for join in joins if len(join[1]) <= degree], dtype=object)
            
        # table names => integers
        joins[:,0] = np.vectorize(int)(joins[:,0])
        
        # create the costs column
        joins = np.hstack([joins, np.zeros((len(joins), 1))])
        
        # get the costs
        for i in range(len(joins[:,0])):
            joins[:,2][i] = np.vectorize(float)(g.plan_details[joins[:,0][i]])

        # table names => integers
        joins[:,0] = np.vectorize(str)(joins[:,0])
        
        # get the corresponding named types from the node labels
        joins[:,0] = np.vectorize(self.labels.get)(joins[:,0])

        # load the canonical node types; match the labels to their canonical symbol
        node_types = load_dict('node_types')
        joins[:,0] = np.vectorize(node_types.get)(joins[:,0])

        # load the canonical table names...
        terminal_dict = load_dict('terminal_dict')
        for i in range(len(joins[:,1])):
            a = np.array(joins[:,1][i][0])
            for j in range(len(joins[:,1][i])):
                joins[:,1][i][j] = terminal_dict[canonical(joins[:,1][i][j])]

        # using the lengths from the global dicts, and np.put, we create the table/type indicator vectors
        # and combine them with the cost
        type_ind = np.zeros((len(joins), len(node_types)))
        table_ind = np.zeros((len(joins), len(terminal_dict)))
        
        # turn the indices into indicator vectors
        for i in range(len(joins)):
            np.put(table_ind[i], joins[:,1][i],1)
            np.put(type_ind[i], joins[:,0][i], 1)
        
        # concat the axis along nx7 ( 7 costs being measured )
        costs = np.concatenate(joins[:,2], axis=0).reshape(len(joins),7)
        return [len(node_types), len(terminal_dict), np.hstack((type_ind, table_ind, costs))]
    


# In[ ]:




