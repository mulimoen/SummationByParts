#! /usr/bin/env python3
import sympy as sp

sp.init_printing(use_unicode=True, use_latex=True)

rho = sp.symbols("rho", real=True, positive=True, nonzero=True)
ru, rv, e = sp.symbols("rhou rhov e", real=True)
gamma = sp.symbols("gamma", real=True, positive=True, nonzero=True)
p = (gamma - 1) * (e - (ru ** 2 + rv ** 2) / (2 * rho))
c = sp.sqrt(gamma * p / rho)

u = ru / rho
v = rv / rho

E = sp.Matrix([rho * u, rho * u * u + p, rho * u * v, u * (e + p)])
F = sp.Matrix([rho * v, rho * u * v, rho * v * v + p, v * (e + p)])

A = E.jacobian([rho, ru, rv, e])
B = F.jacobian([rho, ru, rv, e])

print()
for key in A.eigenvals():
    sp.pprint(key)
print()
for key in B.eigenvals():
    sp.pprint(key)

# Eigenvalues are u, u +- c and v, v +- c
# sp.pprint((key - c).simplify())  # Remainder after subtracting c

# kx, ky = sp.symbols("kx ky")
# Arot = kx*A + ky*B

S, L = A.diagonalize()

sp.pprint(S)
sp.pprint(L)

Labs = (L * L) ** 0.5

plus = S * (L + Labs) * S.inv() / 2
sp.pprint(plus)
minus = S * (L - Labs) * S.inv() / 2
sp.pprint(minus)

S = plus - minus
sp.pprint(S)
