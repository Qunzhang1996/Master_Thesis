from casadi import *

constraint = types.SimpleNamespace()

## Race car parameters
m = 0.043
C1 = 0.5
C2 = 15.5
Cm1 = 0.28
Cm2 = 0.05
Cr0 = 0.011
Cr2 = 0.006

## CasADi Model
# set up states & controls
s = MX.sym("s")
n = MX.sym("n")
alpha = MX.sym("alpha")
v = MX.sym("v")
D = MX.sym("D")
delta = MX.sym("delta")
x = vertcat(s, n, alpha, v, D, delta)# controls
derD = MX.sym("derD")
derDelta = MX.sym("derDelta")
u = vertcat(derD, derDelta)

# dynamics
Fxd = (Cm1 - Cm2 * v) * D - Cr2 * v * v - Cr0 * tanh(5 * v)

a_lat = C2 * v * v * delta + Fxd * sin(C1 * delta) / m
constraint.alat = Function("a_lat", [x, u], [a_lat])
print(constraint.alat)

# Sample state (x) and control (u) values
x_sample = np.array([0.0, 0.0, 0.1, 10.0, 0.2, 0.1])  # Sample state values [s, n, alpha, v, D, delta]
u_sample = np.array([0.1, 0.05])  # Sample control values [derD, derDelta]

# Evaluate the lateral acceleration using the function
a_lat_value = constraint.alat(x_sample, u_sample)

print("Lateral acceleration:", a_lat_value)
