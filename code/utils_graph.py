import networkx as nx


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