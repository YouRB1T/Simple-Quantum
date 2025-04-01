def convert_form_matrix(weights_matrix):
    ages = []
    used_ages = set()
    num_vertices = len(weights_matrix)
    for i in range(num_vertices):
        for j in range(i):
            if i != j and weights_matrix[i][j] != 0 and i not in used_ages and j not in used_ages:
                ages.append([i, j, weights_matrix[i][j]])

    return ages, num_vertices