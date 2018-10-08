
# coding: utf-8

# In[40]:


import networkx as nx
from networkx.algorithms import bipartite as bi
import numpy as np
from graph import *


# In[138]:


class Graphs(Graph):
    def __init__(self):
        super(Graphs, self).__init__()
        self.M = None # homo graph of user
        self.N = None # homo graph of item
        
    def read_edgelist(self, filename, weighted = False, directed = False):
        self.G = nx.DiGraph()
        
        self.edge_dict_u = {}  # edge_dict_u[u][i] = rating
        self.edge_dict_v = {}  # edge_dict_v[i][u] = rating
        self.node_u = [] # u nodes arranged in dict order
        self.node_v = [] # v nodes arranged in dict order
        self.edge_list = [] # [(u, v, weight), (v, u, weight)]
        
        # I only deal with undirected weighted graphs 
        # so here I only modify that part
        if directed:
            def read_unweighted(l):
                src, dst = l.split()
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = 1.0

            def read_weighted(l):
                src, dst, w = l.split()
                self.G.add_edge(src, dst)
                self.G[src][dst]['weight'] = float(w)
        else:
            def read_unweighted(l):
                src, dst = l.split()
                self.G.add_edge(src, dst)
                self.G.add_edge(dst, src)
                self.G[src][dst]['weight'] = 1.0
                self.G[dst][src]['weight'] = 1.0

            def read_weighted(l):
                src, dst, w = l.split()
                self.edge_list.append((src, dst, float(w)))
                self.edge_list.append((dst, src, float(w)))
                
                if src not in self.node_u:
                    self.node_u.append(src)
                if dst not in self.node_v:
                    self.node_v.append(dst)
                
        fin = open(filename, 'r')
        func = read_unweighted
        if weighted:
            func = read_weighted
        while 1:
            l = fin.readline()
            if l == '':
                break
            func(l)
       
        self.G.add_nodes_from(self.node_u, bipartite = 0)
        self.G.add_nodes_from(self.node_v, bipartite = 1)
        self.G.add_weighted_edges_from(self.edge_list)
        fin.close()
        self.encode_node()
    
    def build_homo(self):
        print('Constructing homo graphs....')
        num_u = len(self.node_u)
        self.A = bi.biadjacency_matrix(self.G, self.node_u, self.node_v, dtype = np.float, weight ='weight', format = 'csr')
        
        A_T = self.A.transpose()
        self.mat_u = self.A.dot(A_T)
        self.mat_u.data *= (self.mat_u.data > 3)
      #  self.mat_u.eliminate_zeros()
        
        self.mat_v = A_T.dot(self.A)
        self.mat_v.data *= (self.mat_v.data > 2) 
     #   self.mat_v.eliminate_zeros()
        
        # eliminate self loops
        self.mat_u.setdiag(0)
        self.mat_u.eliminate_zeros()
        self.mat_v.setdiag(0)
        self.mat_v.eliminate_zeros()
        
        print('The homo graph for users.....')
        self.M = nx.from_scipy_sparse_matrix(self.mat_u, create_using = nx.DiGraph())
        
        print('The homo graph for items.....')
        self.N = nx.from_scipy_sparse_matrix(self.mat_v, create_using = nx.DiGraph())

