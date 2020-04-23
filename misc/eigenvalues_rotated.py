#! /usr/bin/env python3
import sympy as sp


A = sp.Matrix([[0, 0, 0], [0, 0, -1], [0, -1, 0]])
B = sp.Matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]])

kx, ky = sp.symbols("kx ky")

A = kx * A + ky * B

print("flux")
sp.pprint(A)

eigenvalues = A.eigenvals()

# sp.pprint(eigenvalues)

eigenvectors = A.eigenvects()

# sp.pprint(eigenvectors)

S, L = A.diagonalize()
Labs = (L*L)**0.5

plus = S*(L + Labs)*S.inv()/2
minus = S*(L - Labs)*S.inv()/2

Ahat = plus + minus


plus.simplify()
minus.simplify()

print("Positive flux")
r = sp.symbols("r")
sp.pprint(plus.subs((kx*kx + ky*ky)**0.5, r))
print("Negative flux")
sp.pprint(minus.subs((kx*kx + ky*ky)**0.5, r))
print("r = ")
sp.pprint((kx*kx + ky*ky)**0.5)


print("Evaluations:")
print("plus (1, 0)")
sp.pprint(plus.subs(kx, 1).subs(ky, 0))
print("plus (0, 1)")
sp.pprint(plus.subs(kx, 0).subs(ky, 1))
print("plus (1/sqrt(2), 1/sqrt(2)")
sp.pprint(plus.subs(kx, 1/2**0.5).subs(ky, 1/2**0.5))

print("positive flux - negative flux")
S = plus - minus
S.simplify
sp.pprint(S.subs((kx*kx + ky*ky)**0.5, r))
