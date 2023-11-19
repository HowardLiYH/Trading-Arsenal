'''
Source: Natural Language Processing with Classification and Vector Spaces, Week 4, Coursera Assignment

Description: K-Nearest Neighbour Implemented with Cosine Similarity
'''



import numpy as np

import numpy as np

def cosine_similarity(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)
    # Calculate the L2 norm of each vector
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)

def nearest_neighbor(v, candidates, k=1):
    """
    Input:
      - v, the vector you are going find the nearest neighbor for
      - candidates: a set of vectors where we will find the neighbors
      - k: top k nearest neighbors to find
    Output:
      - k_idx: the indices of the top k closest vectors in sorted form
    """
    similarity_l = []

    # for each candidate vector...
    for row in candidates:
        # get the cosine similarity
        cos_similarity = cosine_similarity(v, row)

        # append the similarity to the list
        similarity_l.append(cos_similarity)

    # sort the similarity list and get the indices of the sorted list
    # Note: np.argsort sorts in ascending order, we reverse it to descending order
    sorted_ids = np.argsort(similarity_l)[::-1]

    # get the indices of the k most similar candidate vectors
    k_idx = sorted_ids[:k]  # Get the top k indices

    return k_idx


if __name__ == '__main__':
    # Testing nearest_neighbor function
    v = np.array([1, 0, 1])
    candidates = np.array([[1, 0, 5], [-2, 5, 3], [2, 0, 1], [6, -9, 5], [9, 9, 9]])
    print(candidates[nearest_neighbor(v, candidates, 3)])
