'''
Source: Natural Language Processing with Classification and Vector Spaces, Week 3, Coursera Assignment

Description: Euclidean Distance
'''

import numpy as np


def euclidean(A, B):
    """
    Input:
        A: a numpy array which corresponds to a word vector
        B: A numpy array which corresponds to a word vector
    Output:
        d: numerical number representing the Euclidean distance between A and B.
    """

    # euclidean distance

    d = np.linalg.norm(A-B)


    return d
