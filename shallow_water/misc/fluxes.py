#! /usr/bin/env python3
import sympy as sp
from sympy.utilities.codegen import codegen

x, y, t = sp.symbols("x,y,t")
rho, g = sp.symbols("rho,g", positive=True, real=True)
# Ignoring rho, assumed to be constant
eta = sp.Symbol("eta", positive=True, real=True)
etau, etav = sp.symbols("etau,etav", real=True)

u = etau / eta
v = etav / eta
q = sp.Matrix([eta, etau, etav])

E = sp.Matrix([eta * u, eta * u ** 2 + g * eta ** 2 / 2, eta * u * v])
F = sp.Matrix([eta * v, eta * u * v, eta * v ** 2 + g * eta ** 2 / 2])

q = sp.Matrix([eta, etau, etav])
A = E.jacobian(q)
sp.pprint(A)
B = F.jacobian(q)
sp.pprint(B)

f = sp.symbols("f")
coriolis = sp.Matrix([0, -f * v * eta, +f * u * eta])
sp.pprint(coriolis)
b = sp.symbols("b")
frictional = sp.Matrix([0, b * u * eta, b * v * eta])
sp.pprint(frictional)

print("A:")
# We can also diagonalise the transpose,
# which still gives the same (linearised) system.
# This results in a much nicer formulation for A+ and A-
A = A.T
S, D = A.diagonalize()
print("Diagonals:")
sp.pprint(D[0, 0])
sp.pprint(D[1, 1])
sp.pprint(D[2, 2])
print("S:")
sp.pprint(S)
print("SI:")
sp.pprint(sp.simplify(S.inv()))

m = abs(etau / eta) + abs(sp.sqrt(eta ** (3)) * sp.sqrt(g) / eta)

L = D + sp.Matrix([[m, 0, 0], [0, m, 0], [0, 0, m]])
Aplus = sp.simplify(S * L * S.inv()) / 2
Aplus = Aplus.T
L = D - sp.Matrix([[m, 0, 0], [0, m, 0], [0, 0, m]])
Aminus = sp.simplify(S * L * S.inv()) / 2
Aminus = Aminus.T

print("A plus:")
sp.pprint(Aplus)

print("A minus:")
sp.pprint(Aminus)

print("A S:")
sp.pprint(sp.simplify(Aminus - Aplus))

assert sp.simplify(S * D * S.inv() - A) == sp.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

print("B:")
S, D = B.diagonalize()
print("Diagonals:")
sp.pprint(D[0, 0])
sp.pprint(D[1, 1])
sp.pprint(D[2, 2])
print("S:")
sp.pprint(S)
print("SI:")
sp.pprint(sp.simplify(S.inv()))

assert sp.simplify(S * D * S.inv() - B) == sp.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

m = abs(etav / eta) + abs(sp.sqrt(eta ** (3)) * sp.sqrt(g) / eta)

L = D + sp.Matrix([[m, 0, 0], [0, m, 0], [0, 0, m]])
Bplus = sp.simplify(S * L * S.inv()) / 2
L = D - sp.Matrix([[m, 0, 0], [0, m, 0], [0, 0, m]])
Bminus = sp.simplify(S * L * S.inv()) / 2

print("B plus:")
sp.pprint(Bplus)

print("B minus:")
sp.pprint(Bminus)

print("B S:")
sp.pprint(sp.simplify(Bminus - Bplus))

assert sp.simplify((Bplus + Bminus) - B) == sp.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

breakpoint()
code = codegen(
    [("Aplus", Aplus), ("Aminus", Aminus), ("Bplus", Bplus), ("Bminus", Bminus)], "rust"
)
with open("autogen.rs", "w") as f:
    f.write(code[0][1])
