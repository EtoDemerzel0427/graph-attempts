
# coding: utf-8

# In[1]:


import networkx as nx
from networkx.algorithms import bipartite as bi
import numpy as np
import os


# 1. `graph_utils`提供了图的一些操作。其中`construct_bipartite_graph`用于从数据集构造异质图G（每个结点名为数据集中原名）；`construct_homo_graphs`从异质图中构造两个同质图M和N（每个结点名为数字，分别通过`look_up_M`和`look_up_N`两个字典映射回原名（其实也可以通过`node_u`，和`node_v`两个list映射……）。
# 
# 2. 还提供了边采样和负采样的分布表。`gen_node_table`分别产生`user_table`和`item_table`，分别用于用户和物品的点采样（负采样中使用）；边采样实际上没有要求做，因为我们的文件需要罗列每一条边，但以防万一我也写了，可能之后能用上。
# 
# 3. `heter_i`和`heter_u`中罗列了异质图上每一条边，并分别提供了5个负样本（前者为5个物品的负样本，后者为5个用户的负样本）。
# 

# In[2]:


class graph_utils(object):
    def __init__(self, model_path = ''):
        self.model_path = model_path # the path where our data stores
        self.G = nx.Graph() # the bipartite graph
        self.M = nx.Graph() # the homogeneous graph of users
        self.N = nx.Graph() # the homogeneous graph of items
        
        
   
        
        self.edge_dict_u = {}  # edge_dict_u[u][i] = rating
        self.edge_dict_v = {}  # edge_dict_v[i][u] = rating
        self.node_u = [] # u nodes arranged in dict order
        self.node_v = [] # v nodes arranged in dict order
        self.edgelist = [] # [(u, v, weight), (v, u, weight)]
        

        
        self.look_up_M = {}
        self.look_up_N = {}
        
        self.construct_bipartite_graph()
        self.construct_homo_graphs()
        
     #   self.gen_node_table()
      #  self.gen_edge_table()
        
    def construct_bipartite_graph(self):
        '''
        constructing the bipartite graph from an edgelist file.
        
        '''
        
        # get the file name
        filename = os.path.join(self.model_path, 'rating_train.dat')
        
        # test
       # filename = os.path.join(self.model_path, 'edge.txt')
        
        # to store the weighted edge information
        # the element in the list is (u, v, weight) or (v, u , weight)
        edge_list_u_v = []
        edge_list_v_u = []
  #      temp = nx.Graph()
        
        with open(filename, encoding = 'UTF-8') as fin:
            for line in fin:
                user, item, rating = line.strip().split('\t')
                
                # if not in the dict, initialize it
                if self.edge_dict_u.get(user) is None:
                    self.edge_dict_u[user] = {}
                    
                if self.edge_dict_v.get(item) is None:
                    self.edge_dict_v[item] = {}
                
                self.edge_dict_u[user][item] = float(rating)
                self.edge_dict_v[item][user] = float(rating)
                
                # add weighted edges
                edge_list_u_v.append((user, item, float(rating)))
                edge_list_v_u.append((item, user, float(rating)))
                
            
        # create the graph using these data structures
        # arrange the nodes in order
        self.node_u = sorted(list(self.edge_dict_u.keys()))
        self.node_v = sorted(list(self.edge_dict_v.keys()))
        
        self.G.add_nodes_from(self.node_u, bipartite = 0)
        self.G.add_nodes_from(self.node_v, bipartite = 1)
        
        self.G.add_weighted_edges_from(edge_list_u_v + edge_list_v_u)
        
        self.edgelist = edge_list_u_v + edge_list_v_u
        
        #self.G = nx.convert_node_labels_to_integers(temp)
    
    def construct_homo_graphs(self):
        '''
        Split the bipartite graph into 2 homogenous graphs and then combine them as in the BiNE paper.
        
        '''
        print('Constructing homo graphs....')
        num_u = len(self.node_u)
        self.A = bi.biadjacency_matrix(self.G, self.node_u, self.node_v, dtype = np.float, weight ='weight', format = 'csr')
     #   print(A)
        
        A_T = self.A.transpose()
        self.mat_u = self.A.dot(A_T)
        self.mat_v = A_T.dot(self.A)
        
        print('The homo graph for users.....')
        self.M = nx.from_scipy_sparse_matrix(self.mat_u)
        
        print('The homo graph for items.....')
        self.N = nx.from_scipy_sparse_matrix(self.mat_v)
        
        num_M = len(self.node_u)
        num_N = len(self.node_v)
        self.look_up_M = dict(zip(np.arange(num_M), self.node_u))
        self.look_up_N = dict(zip(np.arange(num_N), self.node_v))
        
    def gen_node_table(self):
        table_size = 1e7
        power = 0.75
        
        # generate node table
      #  num_M = len(self.node_u)
       # num_N = len(self.node_v)
        
        node_M_degree = np.array(self.M.degree(weight = 'weight'))[:, 1] # an numpy array
        node_N_degree = np.array(self.N.degree(weight = 'weight'))[:, 1]
        
       
        self.user_table = self.get_distribution(node_M_degree)
        self.item_table = self.get_distribution(node_N_degree)
        
        
    def get_distribution(self, degree_vec):
        table_size = 1e7
        power = 0.75
        
        x = np.power(degree_vec, power)
        norm = np.sum(x)
        print('The norm is', norm)
        
        table = np.zeros(int(table_size), dtype = np.uint32)
        
        
        p = 0
        i = 0
        for j in range(len(degree_vec)):
            p = np.sum(x[0: j + 1])/norm
            # p += np.power(node_degree[j], power) / norm
            while i < table_size and (i / table_size) < p:
                table[i] = j
                i += 1
                
        return table
    
    def gen_edge_table(self):
        
        print('Generating edge table for G...')
        self.prob_G, self.alias_G = self.get_alias_prob(self.G)
        print('Done!')
        
        print('Generating the first table for M...')
        self.prob1_M, self.alias1_M = self.get_alias_prob(self.M)
        print('Done!')
        print('Generating the second table for M...')
        self.prob2_M, self.alias2_M = self.get_alias_prob(self.M, homo = True)
        print('Done!')
        
        print('Generating the first table for N...')
        self.prob1_N, self.alias1_N = self.get_alias_prob(self.N)
        print('Done!')
        print('Generating the second table for N...')
        self.prob2_N, self.alias2_N = self.get_alias_prob(self.N, homo = True)
        print('Done!')

    def get_alias_prob(self, graph, homo = False):
        data_size = graph.number_of_edges()
        total_sum = graph.size(weight = 'weight')
        
        alias = np.zeros(data_size, np.int32)
        prob = np.zeros(data_size, np.float32)
        large_block = np.zeros(data_size, np.int32)
        small_block = np.zeros(data_size, np.int32)
        
       
        if (homo == False):
            norm_prob = [graph[edge[0]][edge[1]]["weight"]*data_size/total_sum for edge in graph.edges()]
        else:
            norm_prob = [graph[edge[1]][edge[0]]["weight"]*data_size/total_sum for edge in graph.edges()]
       
      #  assert(np.sum(norm_prob) == data_size)
        num_small_block = 0
        num_large_block = 0
        cur_small_block = 0
        cur_large_block = 0
        
        for k in range(data_size - 1, -1, -1):
            if norm_prob[k] < 1:
                small_block[num_small_block] = k
       #         print('small',norm_prob[k])
                num_small_block += 1
                
            else:
                large_block[num_large_block] = k
        #        print('large',norm_prob[k])
                num_large_block += 1
                
        #print(norm_prob(small_block))
        #print(norm_prob(large_block))
        while num_small_block and num_large_block:
            num_small_block -= 1
            cur_small_block = small_block[num_small_block]
                        
            num_large_block -= 1
            cur_large_block = large_block[num_large_block]
                        
            prob[cur_small_block] = norm_prob[cur_small_block]
        #    print(prob[cur_small_block])
            alias[cur_small_block] = cur_large_block
           
            norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] -1
            assert(norm_prob[cur_large_block] >= 0)
            
            if norm_prob[cur_large_block] < 1:
                small_block[num_small_block] = cur_large_block
                num_small_block += 1
            else:
                large_block[num_large_block] = cur_large_block
                num_large_block += 1

        while num_large_block:
            num_large_block -= 1
            prob[large_block[num_large_block]] = 1
        while num_small_block:
            num_small_block -= 1
            prob[small_block[num_small_block]] = 1
            
        return prob, alias


# In[3]:


gul = graph_utils()


# In[8]:


def gen_bipartite_samples(gul, filename1, filename2, neg_ratio = 5):
    fout1 = open(filename1, 'w')
    fout2 = open(filename2, 'w')
    
    node_u = np.array(gul.node_u)
    node_v = np.array(gul.node_v)
    
    for edge in gul.G.edges():
        fout1.write('{} {} {}\t\t'.format(edge[0], edge[1], gul.G[edge[0]][edge[1]]['weight']))
        fout2.write('{} {} {}\t\t'.format(edge[1], edge[0], gul.G[edge[0]][edge[1]]['weight']))
        
        # negative sampling
        neg = gul.user_table[np.random.randint(0, 1e7, 5)]
        neg = node_u[neg]
        fout1.write((' '.join(['%s ']*neg.size)+'\n') % tuple(neg))
    
        neg = gul.item_table[np.random.randint(0, 1e7, 5)]
        neg = node_v[neg]
        fout2.write((' '.join(['%s ']*neg.size)+'\n') % tuple(neg))
    fout1.close()
    fout2.close()


# In[14]:


def gen_M_samples(gul, filename, neg_ratio = 5):
    fout = open(filename, 'w')
    
    print('writing data of M...')
    
    node_u = np.array(gul.node_u)
    node_v = np.array(gul.node_v)
    i = 0
    for edge in gul.M.edges():
        i += 1
        fout.write('{} {} {}\t\t'.format(node_u[edge[0]], node_u[edge[1]], gul.M[edge[0]][edge[1]]['weight']))
        neg = gul.user_table[np.random.randint(0, 1e7, 5)]
        neg = node_u[neg]
        fout.write((' '.join(['%s ']*neg.size)+'\n') % tuple(neg))
        
        fout.write('{} {} {}\t\t'.format(node_u[edge[1]], node_u[edge[0]], gul.M[edge[0]][edge[1]]['weight']))
        neg = gul.user_table[np.random.randint(0, 1e7, 5)]
        neg = node_u[neg]
        fout.write((' '.join(['%s ']*neg.size)+'\n') % tuple(neg))
        
        if i % 200000 == 0:
            print('20w done!')
            
    fout.close()


# In[12]:


def gen_N_samples(gul, filename, neg_ratio = 5):
    fout = open(filename, 'w')
    
    print('writing data of N...')
    
    node_u = np.array(gul.node_u)
    node_v = np.array(gul.node_v)
    i = 0
    for edge in gul.N.edges():
        i += 1
        fout.write('{} {} {}\t\t'.format(node_v[edge[0]], node_v[edge[1]], gul.N[edge[0]][edge[1]]['weight']))
        neg = gul.item_table[np.random.randint(0, 1e7, 5)]
        neg = node_v[neg]
        fout.write((' '.join(['%s ']*neg.size)+'\n') % tuple(neg))
        
        fout.write('{} {} {}\t\t'.format(node_v[edge[1]], node_v[edge[0]], gul.N[edge[0]][edge[1]]['weight']))
        neg = gul.item_table[np.random.randint(0, 1e7, 5)]
        neg = node_v[neg]
        fout.write((' '.join(['%s ']*neg.size)+'\n') % tuple(neg))
        

            
    fout.close()

