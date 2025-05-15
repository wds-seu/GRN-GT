import os
from sklearn.preprocessing import scale
import numpy as np
import pandas as pd

class LoadData(object):
    '''given the path of data, return the data format for GNE
    :param path
    return:
     X: a dictionary, ['data_id_list']-- len(links) of id for nodes in links ; 
                      ['data_attr_list']-- a list of attrs for corresponding nodes;
                     ['data_label_list']-- len(links) of neighbor for corresponding nodes

     nodes: a dictionary, ['node_id']--len(nodes) of id for nodes, one by one; ['node_attr']--a list of attrs for corresponding nodes
    '''

    # Three files are needed in the path
    def __init__(self, path, train_links, features_file, normalize_features=True):
        self.path = path
        self.undirected = False
        # Define files to work with
        self.train_links = train_links
        self.datafile = features_file
        self.attrfile = self.path + "data_standard.txt"
        if os.path.exists(self.attrfile):
            os.remove(self.attrfile)

        # Load expression data
        data = pd.read_csv(self.datafile, index_col=0)

        if normalize_features:
            data = pd.DataFrame(scale(data, axis=0))

        data.to_csv(self.attrfile, header=None, sep=' ', mode='a')

        self.node_map = {}
        self.nodes = {}
        self.X = {}

        self.node_neighbors_map = {}  # [nodeid: neighbors_set] each node id maps to its neighbors set
        self.construct_nodes()
        self.read_link()
        self.construct_node_neighbors_map()
        self.construct_X()

    def readExp(self):
        f = open(self.attrfile)
        line = f.readline()
        items = line.strip().split(' ')
        self.attr_M = len(items[1:])
        print("Dimension of attributes:", self.attr_M)

    def construct_nodes(self):
        '''construct the dictionary '''
        self.readExp()
        f = open(self.attrfile)
        i = 0
        self.nodes['node_id'] = []
        self.nodes['node_attr'] = []
        line = f.readline()
        while line:
            line = line.strip().split(' ')
            self.node_map[int(line[0])] = i  # map the node
            self.nodes['node_id'].append(i)  # only put id in nodes, not the original name
            self.nodes['node_attr'].append(line[1:])
            i = i + 1
            line = f.readline()
        f.close()
        self.id_N = i
        print("Number of genes:", self.id_N)

    def construct_X(self):
        self.X['data_id_list'] = np.ndarray(shape=(len(self.links)), dtype=np.int32)
        self.X['data_attr_list'] = np.ndarray(shape=(len(self.links), self.attr_M), dtype=np.float32)
        self.X['data_label_list'] = np.ndarray(shape=(len(self.links), 1), dtype=np.int32)

        for i in range(len(self.links)):
            self.X['data_id_list'][i] = int(self.node_map[int(self.links[i][0])])
            self.X['data_attr_list'][i] = self.nodes['node_attr'][
                int(self.links[i][0])]  # dimension need to change to  self.attr_dim
            self.X['data_label_list'][i, 0] = int(self.node_map[int(self.links[i][1])])  # one neighbor of the node

    def construct_node_neighbors_map(self):
        for link in self.links:
            if self.node_map[int(link[0])] not in self.node_neighbors_map:
                self.node_neighbors_map[self.node_map[link[0]]] = set([self.node_map[int(link[1])]])
            else:
                self.node_neighbors_map[self.node_map[link[0]]].add(self.node_map[int(link[1])])

    def read_link(self):  # read link file to a list of links
        self.links = []
        if self.undirected:
            print("Making adjacency matrix symmetric since the graph is undirected.")
        for edge in self.train_links:
            link = [int(edge[0]), int(edge[1])]
            self.links.append(link)
            if self.undirected:
                link = [int(edge[1]), int(edge[0])]
                self.links.append(link)
