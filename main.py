import snap

def degree(graph):
    # Calulate degree
    degree_count = snap.TIntPrV()
    snap.GetDegCnt(graph, degree_count)


    m = 0
    for item in degree_count:
        # Print the degree of every node
        print("NODE:", item.GetVal1(), "DEGREE:", item.GetVal2())
        m = m + item.GetVal2()
    print("AVG Degree:",(m/len(degree_count)))



if __name__ == '__main__':

    # Load the Graph from Edge List file
    graph = snap.LoadEdgeList(snap.PUNGraph, "datasets/facebook_combined.txt", 0, 1)
    grade(graph)
