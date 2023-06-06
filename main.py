import argparse

import snap
import networkx as nx
import random
import matplotlib.pyplot as plt
import time
import copy
from collections import Counter


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


def graph_to_dict(graph,v_dict,n_dict,t):

    graph_dict = {}
    for node in graph.Nodes():
        graph_dict[node.GetId()] = (v_dict[node.GetId()],n_dict[node.GetId()],t)

    return graph_dict
# graph_dict[key:nodo] = (d+,d-,t)

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

        if random_value <= normalized_probability:
            edges.append((u,v,-1))  # Arco negativo
        else:
            edges.append((u,v,1))  # Arco positivo

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


'''
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
'''

def third_algorithm(graph, k, graph_dict):
    seed_set = set()

    dict = copy.deepcopy(graph_dict)# graph_dict[key:nodo] = (d+,d-,t)

    while len(seed_set) < k:
        calculate_P = lambda node: (dict[node][0] - dict[node][1]) / dict[node][2] if dict[node][0] > dict[node][1] else float('-inf')

        # Trova il nodeID che massimizza P
        max_node = max(dict, key=calculate_P)
        print(max_node)




    return seed_set

def fourth_alhorithm(graph,k,t):
    #comunity -> utilizzare
    print("d")

def comunities_detection(graph):

    # Converti il grafo TNEANet in un grafo PUNGraph (ignorando gli attributi sugli archi)
    G = snap.ConvertGraph(snap.PUNGraph, graph)

    # Esegui l'algoritmo di community detection
    CmtyV = snap.TCnComV()
    modularity = snap.CommunityGirvanNewman(G, CmtyV)

    # Stampa le informazioni sulle community
    print("Numero di community individuate:", len(CmtyV))
    print("Modularità:", modularity)

    # Stampa i nodi appartenenti a ciascuna community
    for i, Cmty in enumerate(CmtyV):
        print("Community", i + 1, ":", )
        for NI in Cmty:
            print(NI, end=" ")
        print()


def TSS_algorithm(graph,k):
    vertices = set([node.GetId() for node in graph.Nodes()])

    seed_set = set()

    while len(seed_set) < k:
        v_with_t_zero = None
        for v in vertices:
            if graph.GetIntAttrDatN(graph.GetNI(v), 't') == 0:
                v_with_t_zero = v
                break

        if v_with_t_zero is not None:  #N(v)   v_with_t_zero --> u
            for edge in graph.Edges():
                u = edge.GetDstNId()
                if u == graph.GetNI(v_with_t_zero):
                    graph.DelEdge(u, v_with_t_zero)
                    graph.AddIntAttrDatN(u, graph.GetIntAttrDatN(u, 't') - 1, 't')
            vertices.remove(v_with_t_zero)
        else:
            max_u = None
            max_value = -1
            v_with_d_less_t = None
            for v in vertices:
                if graph.GetNI(v).GetDeg() < graph.GetIntAttrDatN(v, 't'):
                    v_with_d_less_t = v
                    break

            if v_with_d_less_t is not None:
                seed_set.add(v_with_d_less_t)

                for edge in graph.Edges():
                    u = edge.GetDstNId()
                    if u == graph.GetNI(v_with_d_less_t):
                        graph.DelEdge(u, v_with_d_less_t)
                        graph.AddIntAttrDatN(u, graph.GetIntAttrDatN(u, 't') - 1, 't')
                vertices.remove(v_with_d_less_t)
            else:
                for v in vertices:
                    d = graph.GetNI(v).GetDeg()
                    value = graph.GetIntAttrDatN(v, 't')/( d * (d + 1) )
                    if value > max_value:
                        max_value = value
                        print("value", value, "max_value", max_value, "v", v, "max_u",max_u)
                        max_u = v

                for edge in graph.Edges():
                    u = edge.GetDstNId()
                    src = edge.GetSrcNId()
                    if u == max_u:
                        graph.DelEdge(src, max_u)
                vertices.remove(max_u)

    return seed_set

def cascade(graph,seed_set):

    prev_influenced = set()
    influencing = copy.deepcopy(seed_set)

    while len(influencing) != len(prev_influenced):
        prev_influenced = copy.deepcopy(influencing)
        for node in graph.Nodes():
            nodes_plus = 0
            nodes_minus = 0
            if node.GetId() in prev_influenced:
                continue
            if node.GetId() not in prev_influenced:
                for edge in graph.Edges():
                    u = edge.GetSrcNId()
                    v = edge.GetDstNId()
                    if u == node.GetId() and v in prev_influenced:
                        if graph.GetIntAttrDatE(edge, "weight") == 1:
                            nodes_plus += 1
                        else:
                            nodes_minus += 1
                        print("nodo:", node.GetId(), "plus:", nodes_plus,"minus",nodes_minus)
                if (nodes_plus - nodes_minus) >= graph.GetIntAttrDatN(node,"t"):
                    influencing.add(node.GetId())
    return influencing


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
    filename = "datasets/p2p-Gnutella08.txt"

    parser = argparse.ArgumentParser()

    parser.add_argument('-k', dest='k', action='store',
                        default='', type=int, help='k size of seed-set')
    parser.add_argument('-t', dest='t', action='store',
                        default='', type=int, help='treshold of nodes')
    args = parser.parse_args()


    # Load the Graph from Edge List file
    graph = snap.LoadEdgeList(snap.TUNGraph,filename , 0, 1)

    '''
    random.seed(42)
    graph = snap.TNEANet.New()
    for i in range(1, 50):
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
    
    
    #for edge in graph.Edges():
     #   print(edge.GetSrcNId(), "->", edge.GetDstNId())
    #for node in graph.Nodes():
     #   print(node.GetId())
    '''

    #Labeling
    edges_labeled = labeling(graph)

    v_dict,n_dict = count_degree(edges_labeled)
    print("v",v_dict,"n",n_dict)
    graph_dict = graph_to_dict(graph,v_dict,n_dict,args.t)

    third_algorithm(graph,args.k,graph_dict)




'''
    file_path = "graph.bin"
    FOut = snap.TFOut(file_path)
    graph.Save(FOut)
    FOut.Flush()
    FIn = snap.TFIn(file_path)
    loaded_graph = snap.TNEANet.Load(FIn)
    '''