import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def analyse_erdos_renyi(n, k_mean):
    m = int(k_mean * n / 2)

    graph = nx.gnm_random_graph(n, m)

    degrees = np.zeros(graph.number_of_nodes())
    for node_index, node_degree in nx.degree(graph):
        degrees[node_index] = node_degree

    clusterings = np.zeros(graph.number_of_nodes())
    for node_index, node_clustering in nx.clustering(graph).items():
        clusterings[node_index] = node_clustering

    options = {
        'node_color': 'red',
        'node_size': 2,
        'width': 1
    }
    nx.draw_circular(graph, **options)
    plt.savefig('images/q1_' + str(n) + '_' + str(k_mean) + '_graph.jpg')
    plt.show()

    degree_min = np.min(degrees)
    degree_max = np.max(degrees)
    plt.hist(degrees, bins=np.arange(degree_min - .5, degree_max + 1), density=True)
    plt.xlabel('degree')
    plt.ylabel('density')
    plt.savefig('images/q1_' + str(n) + '_' + str(k_mean) + '_degree.jpg')
    plt.show()

    clustering_max = np.max(clusterings)
    plt.hist(clusterings, range=(0, clustering_max + .1), bins=100, density=True)
    plt.xlabel('clustering')
    plt.ylabel('density')
    plt.savefig('images/q1_' + str(n) + '_' + str(k_mean) + '_clustering.jpg')
    plt.show()


analyse_erdos_renyi(500, .8)
analyse_erdos_renyi(500, 1)
analyse_erdos_renyi(500, 8)
