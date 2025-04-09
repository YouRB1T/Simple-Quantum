import random

import networkx as nx
from matplotlib import pyplot as plt


def convert_form_matrix(weights_matrix):
    ages = []
    used_ages = set()
    num_vertices = len(weights_matrix)
    for i in range(num_vertices):
        for j in range(i):
            if i != j and weights_matrix[i][j] != 0 and i not in used_ages and j not in used_ages:
                ages.append([i, j, weights_matrix[i][j]])

    return ages, num_vertices


def visualize_solution(G, partition):
    layout = nx.spring_layout(G, seed=10)
    node_colors = ['red' if partition[i] == 0 else 'blue' for i in range(len(partition))]
    nx.draw(G, layout, node_color=node_colors, with_labels=True, edge_color="gray")
    labels = {(i, j): G[i][j]['weight'] for i, j in G.edges()}
    nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=labels)
    plt.show()


def visualize_graph(G):
    pos = nx.spring_layout(G)
    if nx.is_weighted(G):
        edge_labels = {k: f'{float(v):.1f}' for k, v in nx.get_edge_attributes(G, 'weight').items()}
        nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', edge_color='gray', font_weight='bold')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, font_color='red')
    else:
        nx.draw(G, with_labels=True, node_size=700, node_color='skyblue', edge_color='gray', font_weight='bold')
    plt.tight_layout()
    plt.show()


def create_weighted_graph(generator_func, *args, weight_range=(1, 10)):
    G = generator_func(*args)
    for u, v in G.edges():
        G[u][v]['weight'] = random.randint(*weight_range)
    return G
