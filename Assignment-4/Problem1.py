#Code for 1a and 1b
import random
import math as m
import numpy as np
import matplotlib.pyplot as plt

N=int(input("Enter the range of N:"))
v=1
sd=m.sqrt(v)
x_values=[]
y_values=[]

while(len(x_values)< N):
  u1=random.uniform(-1,1)
  u2=random.uniform(-1,1)
  s=(u1**2)+(u2**2)
  if(0<s<1):
    k=m.sqrt((-2 * m.log(s)) / s)
    x=u1 * k
    y=u2 * k
    x_values.append(x)
    y_values.append(y)

print("The set of X values:\n",x_values,"\n")
print("The set of Y values:\n",y_values,"\n")

m1=sum(x_values)/len(x_values)
m2=sum(y_values)/len(y_values)
print("The mean of X values:\n",m1,"\n")
print("The mean of Y values:\n",m2,"\n")

D1=[]
D2=[]
for i in range(N):
  d1 = m1 + (sd * x_values[i])
  d2 = m2 + (sd * y_values[i])
  D1.append(d1)
  D2.append(d2)

print("The set of X values:\n",D1,"\n")
print("The set of Y values:\n",D2,"\n")
#code for 1c
m_d1 = sum(D1)/N
m_d2 = sum(D2)/N
print("Mean of D1 is:",m_d1,"\n")
print("Mean of D2 is:",m_d2,"\n")

v_D1 = 0
v_D2 = 0
cv_D12 = 0
for i in range (N):
  v_D1 += (D1[i]-m_d1)**2
  v_D2 += (D2[i]-m_d2)**2
  cv_D12 += (D1[i]-m_d1)*(D2[i]-m_d2)
v_D1 = v_D1/(N-1)
v_D2 = v_D2/(N-1)
cv_D12 = cv_D12/(N-1)
sig=np.array([[v_D1,cv_D12],
 [cv_D12,v_D2]])
print("Covariance Matrix :")
print(sig)

e_values,e_Vectors = np.linalg.eig(sig)
print("\nEigen Values :")
print(e_values)
print("\nEigen Vectors :")
print(e_Vectors)

siginv=np.linalg.inv(sig)
print("Inverse of Covariance Matrix :")
print(siginv)

#Q=(X-mu)^T sigma^-1 (X-mu)
x1=  np.linspace(min(D1)-2, max(D1)+2, 100)
x2= np.linspace(min(D2)-2, max(D2)+2, 100)
X1, X2 = np.meshgrid(x1, x2)

DX1=X1-m_d1
DX2=X2-m_d2

Q=(siginv[0,0]*DX1**2
  + 2*(siginv[0,1]*DX1*DX2)
  + siginv[1,1]*DX2**2)
plt.figure(figsize=(7,6))
levels = [0.5,1,2,3,4,5]
plt.contour(X1,X2,Q,levels=levels)
plt.scatter(D1,D2)
plt.scatter(m_d1, m_d2, s=100)
plt.xlabel("D1")
plt.ylabel("D2")
plt.title("Constant Density Curve")
plt.grid(True)
plt.axis("equal")
plt.show()


eigenvalues, eigenvectors = np.linalg.eig(sig)

# Sort eigenvalues from largest to smallest
order = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

print("Eigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)


# Mean point
center = np.array([m_d1, m_d2])

# Choose a scale for displaying eigenvectors
scale = 2

# Major axis eigenvector
v_major = eigenvectors[:, 0]

# Minor axis eigenvector
v_minor = eigenvectors[:, 1]


# Plot the major axis
plt.plot(
    [center[0] - scale*v_major[0],
     center[0] + scale*v_major[0]],
    [center[1] - scale*v_major[1],
     center[1] + scale*v_major[1]],
    linewidth=2,
    label="Major Axis"
)

# Plot the minor axis
plt.plot(
    [center[0] - scale*v_minor[0],
     center[0] + scale*v_minor[0]],
    [center[1] - scale*v_minor[1],
     center[1] + scale*v_minor[1]],
    linewidth=2,
    label="Minor Axis"
)

# Mean
plt.scatter(
    center[0],
    center[1],
    marker='x',
    s=100,
    label="Mean",
    color="green"
)

plt.legend()
