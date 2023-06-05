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

def labeling_graph(graph, t):
    min_probability = 0  # Valore minimo di probabilità
    max_probability = 1  # Valore massimo di probabilità

    random.seed(time.time())

    edges = []
    for edge in graph.Edges():
        u = edge.GetSrcNId()
        v = edge.GetDstNId()

        #Ciclo da eliminare
        if u == v:
            graph.DelEdge(u, v)
            break

        max_degree = max(graph.GetNI(u).GetDeg(), graph.GetNI(v).GetDeg())

        probability = 1 / (max_degree + 1)
        normalized_probability = (probability - min_probability) / (max_probability - min_probability)

        random_value = random.uniform(0,1)
        #print("u",u,"v",v,"RAND",random_value,"NORM_P",normalized_probability)
        if random_value <= normalized_probability:
            graph.AddIntAttrDatE(edge, -1, "weight")
        else:
            graph.AddIntAttrDatE(edge, 1, "weight")

    return graph

def set_t(graph,t):

    for node in graph.Nodes():
        graph.AddIntAttrDatN(node,t,"t")


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

'''
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
'''


def third_algorithm(graph, k, t):
    seed_set = set()
    # Inizializza il Counter per il conteggio degli archi positivi e negativi
    counter = Counter()

    for edge in graph.Edges():
        src_node_id = edge.GetSrcNId()
        weight = graph.GetIntAttrDatE(edge, "weight")

        if weight > 0:
            # Verifica se il nodo di origine è già presente nel contatore
            if src_node_id in counter:
                # Incrementa il valore 'positive' del contatore del nodo di origine
                counter[src_node_id]['positive'] += 1
            else:
                # Se il nodo di origine non è presente, crea una nuova voce nel contatore
                counter[src_node_id] = {'positive': 1, 'negative': 0}
        elif weight < 0:
            # Verifica se il nodo di origine è già presente nel contatore
            if src_node_id in counter:
                # Incrementa il valore 'positive' del contatore del nodo di origine
                counter[src_node_id]['negative'] += 1
            else:
                # Se il nodo di origine non è presente, crea una nuova voce nel contatore
                counter[src_node_id] = {'positive': 0, 'negative': 1}


    while len(seed_set) < k:
        # Calcola il massimo valore (grado positivo - grado negativo) / t nel Counter se il grado positivo è maggiore del grado negativo
        max_node_id = max(
            (x for x, v in counter.items() if v['positive'] > v['negative']),
            key=lambda x: (counter[x]['positive'] - counter[x]['negative']) / t)
        # Rimuovi l'elemento dal Counter
        counter.pop(max_node_id)
        seed_set.add(max_node_id)

        # Recupera tutti i nodi con max_node_id come vicino
        neighbours = [node.GetId() for node in graph.Nodes() if graph.IsEdge(node.GetId(), max_node_id)]

        # Rimuovi archi positivi o negativi dai vicini
        for neighbour in neighbours:
            edge = graph.GetEI(neighbour, max_node_id)
            weight = graph.GetIntAttrDatE(edge, "weight")

            # Controllo se l'arco è positivo o negativo
            if weight > 0:
                counter[neighbour]['positive'] -= 1
            else:
                counter[neighbour]['negative'] -= 1
        graph.DelNode(max_node_id)  # Rimuovi il nodo dal grafo

    return seed_set




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
                max_u = None
                max_value = -1

                for v in vertices:
                    d = graph.GetNI(v).GetDeg()
                    value = graph.GetIntAttrDatN(v, 't')/( d * (d + 1) )
                    if value > max_value:
                        max_value = value
                        max_u = v

                for edge in graph.Edges():
                    u = edge.GetDstNId()
                    if u == graph.GetNI(v_with_d_less_t):
                        graph.DelEdge(u, max_u)
                vertices.remove(max_u)

    return seed_set

def cascade(graph,seed_set):

    prev_influenced = set()
    influencing = copy.deepcopy(seed_set)

    while len(influencing) != len(prev_influenced):
        nodes_plus = 0
        nodes_minus = 0
        prev_influenced = copy.deepcopy(influencing)
        for node in graph.Nodes():
            if node.GetId() not in prev_influenced:
                for edge in graph.Edges():
                    u = edge.GetSrcNId()
                    v = edge.GetDstNId()
                    if u == node.GetId() and v in prev_influenced:
                        if graph.GetIntAttrDatN(graph.GetNI(u), "weight") == 1:
                            nodes_plus += 1
                        else:
                            nodes_minus += 1
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
    filename = "datasets/twitter_combined.txt"

    parser = argparse.ArgumentParser()

    parser.add_argument('-k', dest='k', action='store',
                        default='', type=int, help='k size of seed-set')
    parser.add_argument('-t', dest='t', action='store',
                        default='', type=int, help='treshold of nodes')
    args = parser.parse_args()


    # Load the Graph from Edge List file
    graph = snap.LoadEdgeList(snap.TNEANet,filename , 0, 1)

    '''
    random.seed(42)
    graph = snap.TNEANet.New()
    for i in range(1, 11):
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
    labeling_graph(graph,args.t)
    set_t(graph,args.t)


    file_path = "graph.bin"
    FOut = snap.TFOut(file_path)
    graph.Save(FOut)
    FOut.Flush()
    FIn = snap.TFIn(file_path)
    loaded_graph = snap.TNEANet.Load(FIn)


    seed_set = TSS_algorithm(loaded_graph,args.k)
    print(seed_set)

    #influenced = cascade(graph, seed_set)
    #print(influenced)
