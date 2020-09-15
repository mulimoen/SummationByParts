#! /usr/bin/env python3
import numpy as np

rstar = 0.5  # Fixed radius
eps = 1.0
M = 0.1  # Indirectly sets p_inf
gamma = 1.4  # Solid

p_inf = 1.0 / (gamma * M * M)
print(f"p_inf: {p_inf}")

dx = 10000.0
dy = 100.0
f = (1 - (dx*dx + dy*dy))/(rstar*rstar)
print(f"f: {f}")

# print(eps*dy/(2*np.pi*np.sqrt(p_inf)*rstar * rstar) * np.exp(f / 2))

u = 1.0 - eps*dy/(2*np.pi*np.sqrt(p_inf)*rstar * rstar) * np.exp(f / 2)
v = 0.0 + eps*dx/(2*np.pi*np.sqrt(p_inf)*rstar * rstar) * np.exp(f / 2)

print(f"sub p: {eps*eps*(gamma - 1)*M*M / (8*np.pi*np.pi*p_inf*rstar*rstar)*np.exp(f)}")


rho = np.power(1.0 - eps*eps*(gamma - 1)*M*M / (
    8*np.pi*np.pi*p_inf*rstar*rstar)*np.exp(f), 1.0/(gamma - 1))
p = (rho**gamma)*p_inf
print(f"p: {p}")
e = p / (gamma - 1) + rho*(u**2 + v**2) / 2

print(f"rho: {v}")
print(f"u: {u}")
print(f"v: {v}")
print(f"e: {e}")
