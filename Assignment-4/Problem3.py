import random
import math as m
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# STEP 1: Generate independent standard Gaussian X1 and X2
# =========================================================

N = int(input("Enter the value of N: "))

x_values = []
y_values = []


while len(x_values) < N:

    u1 = random.uniform(-1, 1)
    u2 = random.uniform(-1, 1)

    s = u1**2 + u2**2

    if 0 < s < 1:

        k = m.sqrt((-2 * m.log(s)) / s)

        x_values.append(u1 * k)


while len(y_values) < N:

    u1 = random.uniform(-1, 1)
    u2 = random.uniform(-1, 1)

    s = u1**2 + u2**2

    if 0 < s < 1:

        k = m.sqrt((-2 * m.log(s)) / s)

        y_values.append(u1 * k)


# =========================================================
# STEP 2: Calculate mean
# =========================================================

mu1 = sum(x_values) / N
mu2 = sum(y_values) / N

mu = np.array([mu1, mu2])

print("\nMean vector:")
print(mu)


# =========================================================
# STEP 3: Calculate original variances
# =========================================================

var_x1 = 0
var_x2 = 0

for i in range(N):

    var_x1 += (x_values[i] - mu1)**2
    var_x2 += (y_values[i] - mu2)**2

var_x1 /= (N - 1)
var_x2 /= (N - 1)

print("\nOriginal variance X1 =", var_x1)
print("Original variance X2 =", var_x2)


# =========================================================
# STEP 4: Calculate base variance
# =========================================================

base_variance = (var_x1 + var_x2) / 2

print("\nBase variance =", base_variance)


# =========================================================
# STEP 5: Create different diagonal variances
# =========================================================

sigma11_sq = 2 * base_variance
sigma22_sq = 0.5 * base_variance


# =========================================================
# STEP 6: Create negative covariance
# =========================================================

rho = -0.7

sigma12 = (
    rho *
    m.sqrt(sigma11_sq * sigma22_sq)
)


print("\nSigma11^2 =", sigma11_sq)
print("Sigma22^2 =", sigma22_sq)
print("Sigma12 =", sigma12)


# =========================================================
# STEP 7: Construct FULL covariance matrix
# =========================================================

Sigma = np.array([
    [sigma11_sq, sigma12],
    [sigma12, sigma22_sq]
])

print("\nFull covariance matrix:")
print(Sigma)


# =========================================================
# STEP 8: Calculate Sigma^(1/2)
# =========================================================

eigenvalues_S, eigenvectors_S = np.linalg.eigh(Sigma)

D_root = np.diag(
    np.sqrt(eigenvalues_S)
)

Sigma_root = (
    eigenvectors_S
    @ D_root
    @ eigenvectors_S.T
)

print("\nSigma^(1/2):")
print(Sigma_root)


# =========================================================
# STEP 9: Form X
# =========================================================

X = np.column_stack(
    (x_values, y_values)
)


# =========================================================
# STEP 10: Transform X
# =========================================================

Y = []

for i in range(N):

    y = mu + np.dot(
        Sigma_root,
        X[i] - mu
    )

    Y.append(y)

Y = np.array(Y)

Y1 = Y[:, 0]
Y2 = Y[:, 1]


# =========================================================
# STEP 11: Calculate mean of Y
# =========================================================

mean_Y1 = sum(Y1) / N
mean_Y2 = sum(Y2) / N

print("\nMean of Y:")
print(mean_Y1, mean_Y2)


# =========================================================
# STEP 12: Calculate covariance of Y
# =========================================================

var_Y1 = 0
var_Y2 = 0
cov_Y = 0

for i in range(N):

    var_Y1 += (Y1[i] - mean_Y1)**2

    var_Y2 += (Y2[i] - mean_Y2)**2

    cov_Y += (
        (Y1[i] - mean_Y1) *
        (Y2[i] - mean_Y2)
    )

var_Y1 /= (N - 1)
var_Y2 /= (N - 1)
cov_Y /= (N - 1)


Sigma_Y = np.array([
    [var_Y1, cov_Y],
    [cov_Y, var_Y2]
])

print("\nEstimated covariance of Y:")
print(Sigma_Y)


# =========================================================
# STEP 13: Eigenvalues and eigenvectors
# =========================================================

eigenvalues, eigenvectors = np.linalg.eigh(Sigma_Y)

order = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)


# =========================================================
# STEP 14: Constant-density curves
# =========================================================

Sigma_inv = np.linalg.inv(Sigma_Y)

x_grid = np.linspace(
    min(Y1) - 1,
    max(Y1) + 1,
    250
)

y_grid = np.linspace(
    min(Y2) - 1,
    max(Y2) + 1,
    250
)

X1, X2 = np.meshgrid(
    x_grid,
    y_grid
)

DX1 = X1 - mean_Y1
DX2 = X2 - mean_Y2

Q = (
    Sigma_inv[0, 0] * DX1**2
    + 2 * Sigma_inv[0, 1] * DX1 * DX2
    + Sigma_inv[1, 1] * DX2**2
)


# =========================================================
# STEP 15: Plot constant-density curves
# =========================================================

plt.figure(figsize=(7, 7))

plt.scatter(
    Y1,
    Y2,
    s=10,
    label="Data"
)

plt.contour(
    X1,
    X2,
    Q,
    levels=[1, 2, 3, 4, 5]
)

plt.scatter(
    mean_Y1,
    mean_Y2,
    marker='x',
    s=100,
    label="Mean"
)

plt.xlabel("Y1")
plt.ylabel("Y2")

plt.title(
    "Problem 3 - Constant Density Curves"
)

plt.grid()
plt.axis("equal")
plt.legend()

plt.show()


# =========================================================
# STEP 16: Plot eigenvectors
# =========================================================

center = np.array([
    mean_Y1,
    mean_Y2
])

major = eigenvectors[:, 0]
minor = eigenvectors[:, 1]

scale = 2

plt.figure(figsize=(7, 7))

plt.scatter(
    Y1,
    Y2,
    s=10,
    label="Data"
)

plt.plot(
    [
        center[0] - scale * major[0],
        center[0] + scale * major[0]
    ],
    [
        center[1] - scale * major[1],
        center[1] + scale * major[1]
    ],
    linewidth=2,
    label="Major Axis"
)

plt.plot(
    [
        center[0] - scale * minor[0],
        center[0] + scale * minor[0]
    ],
    [
        center[1] - scale * minor[1],
        center[1] + scale * minor[1]
    ],
    linewidth=2,
    label="Minor Axis"
)

plt.scatter(
    center[0],
    center[1],
    marker='x',
    s=100,
    label="Mean"
)

plt.xlabel("Y1")
plt.ylabel("Y2")

plt.title(
    "Problem 3 - Eigenvectors and Principal Axes"
)

plt.grid()
plt.axis("equal")
plt.legend()

plt.show()


# =========================================================
# STEP 17: Final results
# =========================================================

print("\n======================================")
print("FINAL RESULTS - PROBLEM 3")
print("======================================")

print("\nMajor eigenvalue =", eigenvalues[0])
print("Minor eigenvalue =", eigenvalues[1])

print("\nMajor eigenvector:")
print(eigenvectors[:, 0])

print("\nMinor eigenvector:")
print(eigenvectors[:, 1])