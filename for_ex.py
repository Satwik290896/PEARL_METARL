import numpy as np

X = []
A = np.array([[2,1],[3,2],[5,6]])
B = np.array([[8,9],[14,17],[90,100]])
for i,(k1,k2) in enumerate(zip(A,B)):
	X = k1
	break
print(X.shape)	
