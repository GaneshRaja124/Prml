import random
import math as m
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# STEP 1: Generate independent standard Gaussian data
# =========================================================

N = int(input("Enter the value of N: "))


def generate_gaussian(N):

    values = []

    while len(values) < N:

        u1 = random.uniform(-1, 1)
        u2 = random.uniform(-1, 1)

        s = u1**2 + u2**2

        if 0 < s < 1:

            k = m.sqrt((-2 * m.log(s)) / s)

            values.append(u1 * k)

    return values


# Generate four independent sets
x11 = generate_gaussian(N)
x12 = generate_gaussian(N)

x21 = generate_gaussian(N)
x22 = generate_gaussian(N)


# =========================================================
# STEP 2: Calculate means of both classes
# =========================================================

mu1_x = sum(x11) / N
mu1_y = sum(x12) / N

mu2_x = sum(x21) / N
mu2_y = sum(x22) / N


mu1 = np.array([
    mu1_x,
    mu1_y
])

mu2 = np.array([
    mu2_x,
    mu2_y
])


print("\nMean of Class 1:")
print(mu1)

print("\nMean of Class 2:")
print(mu2)


# =========================================================
# STEP 3: Calculate variances
# =========================================================

var11 = 0
var12 = 0

var21 = 0
var22 = 0


for i in range(N):

    var11 += (x11[i] - mu1_x)**2
    var12 += (x12[i] - mu1_y)**2

    var21 += (x21[i] - mu2_x)**2
    var22 += (x22[i] - mu2_y)**2


var11 /= (N - 1)
var12 /= (N - 1)

var21 /= (N - 1)
var22 /= (N - 1)


# =========================================================
# STEP 4: Calculate covariance within each class
# =========================================================

cov1 = 0
cov2 = 0


for i in range(N):

    cov1 += (
        (x11[i] - mu1_x) *
        (x12[i] - mu1_y)
    )

    cov2 += (
        (x21[i] - mu2_x) *
        (x22[i] - mu2_y)
    )


cov1 /= (N - 1)
cov2 /= (N - 1)


# =========================================================
# STEP 5: Construct covariance matrices
# =========================================================

Sigma1 = np.array([
    [var11, cov1],
    [cov1, var12]
])


Sigma2 = np.array([
    [var21, cov2],
    [cov2, var22]
])


print("\nCovariance matrix of Class 1:")
print(Sigma1)

print("\nCovariance matrix of Class 2:")
print(Sigma2)


# =========================================================
# STEP 6: To make the two classes visibly different,
# shift their means
# =========================================================

# Use the calculated spread to determine the shift

spread1 = m.sqrt((var11 + var12) / 2)
spread2 = m.sqrt((var21 + var22) / 2)

mu1_final = np.array([
    -2 * spread1,
    0
])

mu2_final = np.array([
    2 * spread2,
    0
])


print("\nFinal Class 1 mean:")
print(mu1_final)

print("\nFinal Class 2 mean:")
print(mu2_final)


# =========================================================
# STEP 7: Calculate Sigma^(1/2) for Class 1
# =========================================================

eigenvalues1, eigenvectors1 = np.linalg.eigh(Sigma1)

D1_root = np.diag(
    np.sqrt(eigenvalues1)
)

Sigma1_root = (
    eigenvectors1
    @ D1_root
    @ eigenvectors1.T
)


# =========================================================
# STEP 8: Calculate Sigma^(1/2) for Class 2
# =========================================================

eigenvalues2, eigenvectors2 = np.linalg.eigh(Sigma2)

D2_root = np.diag(
    np.sqrt(eigenvalues2)
)

Sigma2_root = (
    eigenvectors2
    @ D2_root
    @ eigenvectors2.T
)


# =========================================================
# STEP 9: Generate Class 1
# =========================================================

X1 = np.column_stack(
    (x11, x12)
)

Y1 = []

for i in range(N):

    y = (
        mu1_final
        + np.dot(
            Sigma1_root,
            X1[i] - np.array([mu1_x, mu1_y])
        )
    )

    Y1.append(y)


Y1 = np.array(Y1)


# =========================================================
# STEP 10: Generate Class 2
# =========================================================

X2 = np.column_stack(
    (x21, x22)
)

Y2 = []

for i in range(N):

    y = (
        mu2_final
        + np.dot(
            Sigma2_root,
            X2[i] - np.array([mu2_x, mu2_y])
        )
    )

    Y2.append(y)


Y2 = np.array(Y2)


# =========================================================
# STEP 11: Separate coordinates
# =========================================================

Y1_x = Y1[:, 0]
Y1_y = Y1[:, 1]

Y2_x = Y2[:, 0]
Y2_y = Y2[:, 1]


# =========================================================
# STEP 12: Calculate actual means
# =========================================================

mean1_x = sum(Y1_x) / N
mean1_y = sum(Y1_y) / N

mean2_x = sum(Y2_x) / N
mean2_y = sum(Y2_y) / N


print("\nActual Class 1 mean:")
print(mean1_x, mean1_y)

print("\nActual Class 2 mean:")
print(mean2_x, mean2_y)


# =========================================================
# STEP 13: Calculate covariance of Class 1
# =========================================================

v11 = 0
v12 = 0
c12 = 0


for i in range(N):

    v11 += (Y1_x[i] - mean1_x)**2

    v12 += (Y1_y[i] - mean1_y)**2

    c12 += (
        (Y1_x[i] - mean1_x) *
        (Y1_y[i] - mean1_y)
    )


v11 /= (N - 1)
v12 /= (N - 1)
c12 /= (N - 1)


Cov1 = np.array([
    [v11, c12],
    [c12, v12]
])


# =========================================================
# STEP 14: Calculate covariance of Class 2
# =========================================================

v21 = 0
v22 = 0
c22 = 0


for i in range(N):

    v21 += (Y2_x[i] - mean2_x)**2

    v22 += (Y2_y[i] - mean2_y)**2

    c22 += (
        (Y2_x[i] - mean2_x) *
        (Y2_y[i] - mean2_y)
    )


v21 /= (N - 1)
v22 /= (N - 1)
c22 /= (N - 1)


Cov2 = np.array([
    [v21, c22],
    [c22, v22]
])


print("\nEstimated covariance Class 1:")
print(Cov1)

print("\nEstimated covariance Class 2:")
print(Cov2)


# =========================================================
# STEP 15: Plot the two classes
# =========================================================

plt.figure(figsize=(8, 7))

plt.scatter(
    Y1_x,
    Y1_y,
    s=10,
    label="Class 1"
)

plt.scatter(
    Y2_x,
    Y2_y,
    s=10,
    label="Class 2"
)

plt.scatter(
    mean1_x,
    mean1_y,
    marker='x',
    s=100,
    color='red'
)

plt.scatter(
    mean2_x,
    mean2_y,
    marker='x',
    s=100,
    color='blue'
)

plt.xlabel("X1")
plt.ylabel("X2")

plt.title(
    "Problem 4 - Two Gaussian Classes"
)

plt.grid()
plt.axis("equal")
plt.legend()

plt.show()


# =========================================================
# STEP 16: Gaussian discriminant functions
#
# g_i(x) =
# -1/2 ln|Sigma_i|
# -1/2 (x-mu_i)^T Sigma_i^(-1)(x-mu_i)
#
# Equal priors are assumed.
# =========================================================

inv1 = np.linalg.inv(Cov1)
inv2 = np.linalg.inv(Cov2)

det1 = np.linalg.det(Cov1)
det2 = np.linalg.det(Cov2)


def discriminant(
    x,
    y,
    mu,
    covariance,
    inverse,
    determinant
):

    dx = x - mu[0]
    dy = y - mu[1]

    vector = np.array([
        dx,
        dy
    ])

    distance = np.dot(
        vector,
        np.dot(inverse, vector)
    )

    return (
        -0.5 * m.log(determinant)
        -0.5 * distance
    )


# =========================================================
# STEP 17: Create grid
# =========================================================

xmin = min(
    min(Y1_x),
    min(Y2_x)
) - 1

xmax = max(
    max(Y1_x),
    max(Y2_x)
) + 1

ymin = min(
    min(Y1_y),
    min(Y2_y)
) - 1

ymax = max(
    max(Y1_y),
    max(Y2_y)
) + 1


x_grid = np.linspace(
    xmin,
    xmax,
    300
)

y_grid = np.linspace(
    ymin,
    ymax,
    300
)

XX, YY = np.meshgrid(
    x_grid,
    y_grid
)


# =========================================================
# STEP 18: Calculate discriminant difference
# =========================================================

G1 = np.zeros_like(XX)
G2 = np.zeros_like(XX)


for i in range(len(y_grid)):

    for j in range(len(x_grid)):

        G1[i, j] = discriminant(
            XX[i, j],
            YY[i, j],
            np.array([mean1_x, mean1_y]),
            Cov1,
            inv1,
            det1
        )

        G2[i, j] = discriminant(
            XX[i, j],
            YY[i, j],
            np.array([mean2_x, mean2_y]),
            Cov2,
            inv2,
            det2
        )


# =========================================================
# STEP 19: Plot decision boundary
# =========================================================

plt.figure(figsize=(8, 7))

plt.scatter(
    Y1_x,
    Y1_y,
    s=10,
    label="Class 1"
)

plt.scatter(
    Y2_x,
    Y2_y,
    s=10,
    label="Class 2"
)


# Decision boundary:
# g1(x) - g2(x) = 0

plt.contour(
    XX,
    YY,
    G1 - G2,
    levels=[0],
    linewidths=2
)


plt.scatter(
    mean1_x,
    mean1_y,
    marker='x',
    s=100,
    color='red'
)

plt.scatter(
    mean2_x,
    mean2_y,
    marker='x',
    s=100,
    color='blue'
)


plt.xlabel("X1")
plt.ylabel("X2")

plt.title(
    "Problem 4 - Decision Boundary"
)

plt.grid()
plt.axis("equal")
plt.legend()

plt.show()


# =========================================================
# STEP 20: Final output
# =========================================================

print("\n======================================")
print("PROBLEM 4 FINAL RESULTS")
print("======================================")

print("\nClass 1 mean:")
print([mean1_x, mean1_y])

print("\nClass 2 mean:")
print([mean2_x, mean2_y])

print("\nClass 1 covariance:")
print(Cov1)

print("\nClass 2 covariance:")
print(Cov2)