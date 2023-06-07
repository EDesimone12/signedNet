import argparse

import snap
import networkx as nx
import random
import matplotlib.pyplot as plt
import time
import copy
from collections import Counter

#Crea i pesi degli archi
def labeling(graph, t):
    min_probability = 0  # Valore minimo di probabilità
    max_probability = 1  # Valore massimo di probabilità

    random.seed(time.time())
    
    #Aggiungo anche il t sui nodi
    for node in graph.Nodes():
        graph.AddIntAttrDatN(node, t, "t")
        graph.AddIntAttrDatN(node, t, "t")

    for edge in graph.Edges():

        u = edge.GetSrcNId()
        v = edge.GetDstNId()
        
        if u == v:
            graph.DelEdge(u, v)
            break
        
        max_degree = max(graph.GetNI(u).GetDeg(), graph.GetNI(v).GetDeg())

        probability = 1 / (max_degree + 1)
        normalized_probability = (probability - min_probability) / (max_probability - min_probability)

        random_value = random.uniform(0,1)

        if random_value <= normalized_probability:
            graph.AddIntAttrDatE(edge, -1, 'weight')
        else:
            graph.AddIntAttrDatE(edge, 1, 'weight')  # Arco positivo

#Recupera i nodi adiacenti (sia in che out)
def get_neighbors(graph, id_node):
    neighbors = set()
    node = graph.GetNI(id_node)
    
    for i in range(node.GetDeg()):
        neighbor = node.GetNbrNId(i)
        neighbors.add(neighbor)
    
    return neighbors

#Setta i gradi sia positivi che negativi dei nodi
def set_degree_node(graph):
    
    #Inizialmente i gradi per tutti i nodi vengono messi a 0 altrimenti non è possibile leggere l'attributo
    for node in graph.Nodes():
        graph.AddIntAttrDatN(node, 0, "degree_pos")
        graph.AddIntAttrDatN(node, 0, "degree_neg")
        graph.AddIntAttrDatN(node, 0, 'degree_tot')
    
    for edge in graph.Edges():
        src_id = edge.GetSrcNId()
        dst_id = edge.GetDstNId()
        
        src_node = graph.GetNI(src_id)
        dst_node = graph.GetNI(dst_id)
        
        #Setto il grado generale dei nodi
        graph.AddIntAttrDatN(src_node, graph.GetIntAttrDatN(src_node, 'degree_tot') + 1, 'degree_tot')
        graph.AddIntAttrDatN(dst_node, graph.GetIntAttrDatN(src_node, 'degree_tot') + 1, 'degree_tot')
        
        #Sia della sorgente che della destinazione, perché siamo su un grafo orientato
        if(graph.GetIntAttrDatE(edge, "weight") == 1):
            graph.AddIntAttrDatN(src_node, graph.GetIntAttrDatN(src_node, 'degree_pos') + 1, 'degree_pos')
            graph.AddIntAttrDatN(src_node, graph.GetIntAttrDatN(dst_node, 'degree_pos') + 1, 'degree_pos')
        else: 
            graph.AddIntAttrDatN(src_node, graph.GetIntAttrDatN(src_node, 'degree_neg') + 1, 'degree_neg')
            graph.AddIntAttrDatN(src_node, graph.GetIntAttrDatN(dst_node, 'degree_neg') + 1, 'degree_neg')

#Recupera i vicini positivi o negativi di un nodo
def get_neighbors_weighed(graph, node_id, weight):
    neighbors = get_neighbors(graph, node_id)
    new_neighbors = set()
    
    for neighbor_id in neighbors:
        edge = None
        #Recupero l'arco corretto (perché è un grafo non direzionato, ma si comporta come direzionato)
        if graph.GetNI(neighbor_id).IsOutNId(node_id):
            edge = graph.GetEI(neighbor_id, node_id) #Arco tra max_node_id -> neighbor_id
        else: 
            edge = graph.GetEI(node_id, neighbor_id) #Arco tra neighbor_id -> max_node_id

        weight_edge = graph.GetIntAttrDatE(edge, "weight")
        
        if weight_edge == weight:
            new_neighbors.add(neighbor_id)
            
    return new_neighbors

def third_algorithm(graph, k):
    counter = Counter()
    seed_set = set()
            
    while len(seed_set) < k:
        
        #Metto nel counter gli elementi che rispettano la condizione degree_pos >= degree_neg
        #Lo ricalcolo ogni volta perché il grado può essere decrementato
        for node in graph.Nodes():
            node_id = node.GetId()
            degree_pos = graph.GetIntAttrDatN(node_id, 'degree_pos')
            degree_neg = graph.GetIntAttrDatN(node_id, 'degree_neg')
            t = graph.GetIntAttrDatN(node_id, 't')
            
            if degree_pos >= degree_neg:
                counter[node_id] = degree_pos - degree_neg / t
        
        max_node_id = max(counter, key=counter.get) # Ottieni la chiave con il valore massimo
        counter.clear() # Svuoto il Counter
        seed_set.add(max_node_id) #Aggiungo al seed_set
        
        #Prendo la lista dei nodi adiacenti al nodo aggiungo al seed_set
        neighbors = get_neighbors(graph, max_node_id)
        for neighbor_id in neighbors:
            
            edge = None
            #Recupero l'arco corretto (perché è un grafo non direzionato, ma si comporta come direzionato)
            if graph.GetNI(neighbor_id).IsOutNId(max_node_id):
                edge = graph.GetEI(neighbor_id, max_node_id) #Arco tra max_node_id -> neighbor_id
            else: 
                edge = graph.GetEI(max_node_id, neighbor_id) #Arco tra neighbor_id -> max_node_id
            
            
            weight = graph.GetIntAttrDatE(edge, "weight")
            
            if weight == 1: #Decremente il grado positivo sia di max_node_id che di neighbor_id
                graph.AddIntAttrDatN(max_node_id, graph.GetIntAttrDatN(max_node_id, 'degree_pos') - 1, 'degree_pos')
                graph.AddIntAttrDatN(neighbor_id, graph.GetIntAttrDatN(neighbor_id, 'degree_pos') - 1, 'degree_pos')
            else: #Decremente il grado negativo sia di max_node_id che di neighbor_id
                graph.AddIntAttrDatN(max_node_id, graph.GetIntAttrDatN(max_node_id, 'degree_neg') - 1, 'degree_neg')
                graph.AddIntAttrDatN(neighbor_id, graph.GetIntAttrDatN(neighbor_id, 'degree_neg') - 1, 'degree_neg')
    
    return seed_set

def sup_TSS(graph, dict_neighbor, node_id, flag_t = True):
    neighbors = get_neighbors(graph, node_id) #Recupero i vicini di node_t_with_zero
            
    #Per ogni vicino decremento la t e il grado
    for neighbor in neighbors:
        
        if flag_t:
            graph.AddIntAttrDatN(
                neighbor, 
                graph.GetIntAttrDatN(neighbor, 't') - 1, 
                't')
        
        graph.AddIntAttrDatN(
            neighbor, 
            graph.GetIntAttrDatN(node_id, 'degree_tot') - 1, 
            'degree_tot')
        
        #Elimino v dai vicini di u
        new_neighbors = set(dict_neighbor[neighbor])
        new_neighbors.discard(node_id)
        dict_neighbor[neighbor] = new_neighbors
    
def TSS(graph, k): 
    seed_set = set()
    dict_neighbor = {}
    counter = Counter()
    v = None
    
    for node in graph.Nodes():
        neighbors = get_neighbors(graph, node.GetId())            
        dict_neighbor[node.GetId()] = copy.deepcopy(neighbors)
    
    while len(seed_set) < k:
        node_t_with_zero = None
        node_t_degree_min_t = None
        
        #Trovo nodo con t == 0
        for node in graph.Nodes():
            node_id = node.GetId()
            if graph.GetIntAttrDatN(node_id, 't') == 0:
                node_t_with_zero = node_id
                break
            
        #Trovo nodo con d < t
        for node in graph.Nodes():
            node_id = node.GetId()
            if graph.GetIntAttrDatN(node_id, 'degree_tot') < graph.GetIntAttrDatN(node_id, 't'):
                node_t_degree_min_t = node_id
                break
        
        #Se esiste un nodo con t = 0
        if node_t_with_zero is not None:
            sup_TSS(graph, dict_neighbor, node_t_with_zero)
            v =  node_t_with_zero           
        else:
            if node_t_degree_min_t is not None:
                seed_set.add(node_t_degree_min_t)
                sup_TSS(graph, dict_neighbor, node_t_degree_min_t)
                
                v = node_t_degree_min_t
            else:
                for node in graph.Nodes():
                    node_id = node.GetId()
                    degree = graph.GetIntAttrDatN(node_id, 'degree_tot')
                    t = graph.GetIntAttrDatN(node_id, 't')
                    
                    counter[node_id] = t / degree * (degree + 1)
                
                max_node_id = max(counter, key=counter.get) # Ottieni la chiave con il valore massimo
                counter.clear() # Svuoto il Counter
                v = max_node_id
                
                sup_TSS(graph, dict_neighbor, max_node_id, False)
                
        graph.DelNode(v)
        del dict_neighbor[v]
    
    return seed_set
                 
def cascade(graph, seed_set): 
    prev_influenced = set()
    influencing = copy.deepcopy(seed_set)

    while len(influencing) != len(prev_influenced):
        prev_influenced = copy.deepcopy(influencing)
        
        for node in graph.Nodes():
            node_id = node.GetId()
            
            #Condizione di guarda per prendere nodi non influenzati
            if node_id in prev_influenced:
                continue
        
            neighbors_pos = get_neighbors_weighed(graph, node_id, 1)
            neighbors_neg = get_neighbors_weighed(graph, node_id, -1)
            
            intersection_set_pos = neighbors_pos.intersection(prev_influenced) #Calcolo l'intersezione di neighbors_pos con prev_influenzed
            
            intersection_set_neg = neighbors_neg.intersection(prev_influenced) #Calcolo l'intersezione di neighbors_neg con prev_influenzed
            
            t = graph.GetIntAttrDatN(node_id, 't')
            
            if len(intersection_set_pos) - len(intersection_set_neg) >= t:
                influencing.add(node_id)
                
    return influencing

def nostro(graph, k):
    seed_set = set()

    G = snap.ConvertGraph(snap.PUNGraph, graph)

    # Esegui l'algoritmo di community detection
    CmtyV = snap.TCnComV()
    modularity = snap.CommunityCNM(G, CmtyV)

    # Stampa le informazioni sulle community
    print("Numero di community individuate:", len(CmtyV))
    print("Modularità:", modularity)

    # Stampa i nodi appartenenti a ciascuna community
    Cmty_list = []
    for i, Cmty in enumerate(CmtyV):
        #print("Community", i + 1, ":", )
        Cmty_list.append((i,Cmty,len(Cmty)))
        #for NI in Cmty:
        #    print(NI, end=" ")
        #print()

    ordered_Cmty = sorted(Cmty_list, key=lambda x: x[2], reverse=True)
    #print(ordered_Cmty)

    #More communities than k
    final_cmty = []
    if len(ordered_Cmty) > k:
        for Cmt in ordered_Cmty:
            final_cmty.append((Cmt[0],Cmt[1]))
        '''
        print("Final") #Just printing the final Cmty list
        for tupl_Cmty in final_cmty:
            print("Community", tupl_Cmty[0], ":", )
            for NI in tupl_Cmty[1]:
                print(NI, end=" ")
            print()
        '''
        for tupl_Cmty in final_cmty:
            NIdV = snap.TIntV()
            for NI in tupl_Cmty[1]:
                NIdV.Add(NI)
            SubGraph = snap.GetSubGraph(graph, NIdV)

            # Calcola la centralità di intermediazione per la community corrente
            CentrH, _ = snap.GetBetweennessCentr(SubGraph)

            # Stampa la centralità di intermediazione per ogni nodo nella community
            #print("Community", tupl_Cmty[0], ":",end="")
            #for NodeId in CentrH:
                #print("Nodo", NodeId, ":", CentrH[NodeId])
            # print("Nodo_max",max_centr_node,"value:",CentrH[max_centr_node])
            new_node = None
            while(new_node is None):
                max_centr_node = max(CentrH, key=CentrH.get)
                deg_neg = graph.GetIntAttrDatN(max_centr_node, 'degree_neg')
                deg_pos = graph.GetIntAttrDatN(max_centr_node, 'degree_pos')
                if (deg_pos - deg_neg) >= graph.GetIntAttrDatN(max_centr_node, 't'):
                    new_node = max_centr_node
                else:
                    del CentrH[max_centr_node]
            seed_set.add(new_node)


    else:
        #less communities than k, so k/n_Cmty elements of the seed_set for every Cmty
        print("else")

    
if __name__ == '__main__':
    filename = "datasets/facebook_combined.txt"

    parser = argparse.ArgumentParser()

    parser.add_argument('-k', dest='k', action='store',
                        default='', type=int, help='k size of seed-set')
    parser.add_argument('-t', dest='t', action='store',
                        default='', type=int, help='treshold of nodes')
    args = parser.parse_args()


    # Load the Graph from Edge List file
    graph = snap.LoadEdgeList(snap.TNEANet,filename , 0, 1)

    #Labeling
    labeling(graph, args.t)
    
    #Setto i gradi positivi e negativi su ogni nodo
    set_degree_node(graph)
    
    #Salvo il grafo solo con label e attributi, in modo da passare al cascade un grafo senza manipolazioni
    file_path = "graph.bin"
    FOut = snap.TFOut(file_path)
    graph.Save(FOut)
    FOut.Flush()
    FIn = snap.TFIn(file_path)
    loaded_graph = snap.TNEANet.Load(FIn)

    #seed_set = third_algorithm(graph, args.k)
    #seed_set = TSS(graph, args.k)
    '''seed_set = set({4, 713, 9, 107, 11, 3437, 12, 15, 18, 693})
    print(seed_set)
    print(f"Lunghezza seed_set: {len(seed_set)}")
    
    influenced = cascade(loaded_graph, seed_set)
    print(influenced)
    print(f"Lunghezza influenced: {len(influenced)}")'''
    nostro(graph,args.k)