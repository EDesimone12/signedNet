import argparse

import snap
import networkx as nx
import random
import matplotlib.pyplot as plt
import scipy as sp
import time


def count_degree(edges_labeled):
    v_dict = {} #defining a dictionary of key:nodeID and value: outDegree  d^+
    n_dict = {} #same usage, d^-

    for e in edges_labeled: #(u,v,peso)
        if e[2] == 1:
            if e[0] in v_dict:
                v_dict[e[0]] = v_dict[e[0]] + 1 #d^+ degree of a node
            else:
                v_dict[e[0]] = 1
        if e[2] == -1:
            if e[0] in n_dict:
                n_dict[e[0]] = n_dict[e[0]] + 1 #d^+ degree of a node
            else:
                n_dict[e[0]] = 1

    return v_dict,n_dict


def get_Neighbours_ofID(edges_labeled,v_dict,nodeID, sign_value):
    #this function provides a list of all the nodes that have an edgeOut in nodeID
    neighbours = []

    for e in edges_labeled:
        if(e[1] == nodeID and e[2] == sign_value and e[0] in v_dict):
            neighbours.append(e[0])

    return neighbours

def compact_dict(v_dict,n_dict):
    #[[u,d+,d-]...]  <---> dict[key:u] = (d+,d-)
    comp_dict = {}
    for v in v_dict.keys():
        comp_dict[v] = (v,v_dict[v],0)
    for v in n_dict.keys():
        if v in comp_dict:
            tupla = comp_dict[v]
            comp_dict[v] = (v,tupla[0],n_dict[v])
        else:
            comp_dict[v] = (v, 0 ,n_dict[v])
    return comp_dict

def labeling(graph):
    min_probability = 0  # Valore minimo di probabilità
    max_probability = 1  # Valore massimo di probabilità

    random.seed(time.time())

    edges = []
    for edge in graph.Edges():

        u = edge.GetSrcNId()
        v = edge.GetDstNId()
        max_degree = max(graph.GetNI(u).GetDeg(), graph.GetNI(v).GetDeg())

        probability = 1 / (max_degree + 1)
        normalized_probability = (probability - min_probability) / (max_probability - min_probability)

        random_value = random.uniform(0,1)
        #print("u",u,"v",v,"RAND",random_value,"NORM_P",normalized_probability)
        if random_value <= normalized_probability:
            edges.append((u, v, -1))  # Arco negativo
        else:
            edges.append((u, v, 1))  # Arco positivo

    return edges


def first_algorithm(edges_labeled,k):
    #first algorithm - Seeds Greedy Degree max
    seed_set = set()

    v_dict, _ = count_degree(edges_labeled)

    while len(seed_set) < k:
        max_key = max(v_dict, key=v_dict.get)

        seed_set.add(max_key) # Add the new v in the seed set
        del v_dict[max_key] # Removing this v from the dict

    return seed_set


def second_algorithm(edges_labeled,k):
    # second algorithm Seeds Greedy Residual Degree Max

    seed_set = set()

    v_dict, _ = count_degree(edges_labeled)
    # u nel seedset, se w->u allora il grado di u deve essere diminuito

    while len(seed_set) < k:
        max_ID = max(v_dict, key=v_dict.get)
        seed_set.add(max_ID) # Add the new v in the seed set
        del v_dict[max_ID] # Removing this v from the dict

        max_ID_neighbours = get_Neighbours_ofID(edges_labeled,v_dict,max_ID,1)

        for w in max_ID_neighbours: #we have to reduce the degree of the neighbour of max_ID
                v_dict[w] = v_dict[w] - 1

    return seed_set

def third_algorithm(edges_labeled,k,t):
    seed_set = set()

    #v_dict = + DEGREE   - n_dict = - DEGREE
    v_dict, n_dict = count_degree(edges_labeled)
    comp_dict = compact_dict(v_dict,n_dict)

    diff = 0
    while len(seed_set) < k:
        difference_dict = {}
        for e in edges_labeled: #e(u,v,peso)   comp[key:u] = (d+,d-)
            if e[0] not in comp_dict: break
            tupla = comp_dict[e[0]]
            diff = tupla[0] - tupla[1]

            if diff >= 0:
                tot = diff / t
                difference_dict[e[0]] = tot

        if len(difference_dict) == 0:
            return "ERROR DIFF EMPTY"

        max_ID = max(difference_dict, key=difference_dict.get)
        seed_set.add(max_ID) # Add the new v in the seed set
        del comp_dict[max_ID]

        for e in edges_labeled:

            if e[0] not in comp_dict: break
            tupla = comp_dict[e[0]]

            if e[1] == max_ID:
                if e[2] == 1:
                    comp_dict[e[0]] = (tupla[0] - 1,tupla[1])
                else:
                    comp_dict[e[0]] = (tupla[0] , tupla[1] - 1)
        #w vicini di maxID che hanno archi positivi verso maxID
        #z vicini di maxID he hanno archi negativi verso maxID
        # we have to reduce the degree of the neighbour of max_ID

    return seed_set

def printGr(filename):
    G = nx.Graph()

    with open(filename, 'r') as file:
        for line in file:
            source, target = line.strip().split()
            G.add_edge(source, target)

    # Creazione del grafico
    pos = nx.spring_layout(G)  # Layout del grafico
    nx.draw(G, pos, with_labels=False, node_color='blue', node_size = 30)

    plt.show()

if __name__ == '__main__':
    filename = "datasets/facebook_combined.txt"

    parser = argparse.ArgumentParser()

    parser.add_argument('-k', dest='k', action='store',
                        default='', type=int, help='k size of seed-set')
    parser.add_argument('-t', dest='t', action='store',
                        default='', type=int, help='treshold of nodes')
    args = parser.parse_args()


    # Load the Graph from Edge List file
    graph = snap.LoadEdgeList(snap.TNGraph,filename , 0, 1)

    '''
    random.seed(42)
    graph = snap.TNGraph.New()
    for i in range(1, 110):
        graph.AddNode(i)

    for u in graph.Nodes():
        for v in graph.Nodes():
            if u.GetId() != v.GetId():
                # Genera un valore casuale tra 0 e 1
                random_value = random.random()
                if random_value <= 0.5:
                    graph.AddEdge(u.GetId(), v.GetId())
    print("Numero di nodi:", graph.GetNodes())
    print("Numero di archi:", graph.GetEdges())
    '''
    
    #for edge in graph.Edges():
     #   print(edge.GetSrcNId(), "->", edge.GetDstNId())
    #for node in graph.Nodes():
     #   print(node.GetId())



    #Labeling
    edges_labeled = labeling(graph)

    seed_set = third_algorithm(edges_labeled,args.k,args.t)
    print(seed_set)







