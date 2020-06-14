#! /usr/bin/env python3
import numpy as np

A = np.array([[0, 0, 0], [0, 0, -1], [0, -1, 0]])
B = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])


def similarity_transform(matrix):
    L, S = np.linalg.eig(matrix)
    L = np.diag(L)
    S = S.transpose()
    assert np.allclose(np.matmul(S.transpose(), np.matmul(L, S)), matrix)
    return L, S


def plusminus(matrix):
    L, S = similarity_transform(matrix)

    def signed(op):
        return 0.5 * np.matmul(S.transpose(), np.matmul(op(L, np.abs(L)), S))

    plus = signed(np.add)
    minus = signed(np.subtract)

    assert np.allclose(matrix, plus + minus)
    return plus, minus


Aplus, Aminus = plusminus(A)
Bplus, Bminus = plusminus(B)

print("A+")
print(Aplus)
print("A-")
print(Aminus)
print()
print("B+")
print(Bplus)
print("B-")
print(Bminus)
