import os
import matplotlib.pyplot as plt
import sys

#print(sys.argv[1])
os.chdir("./" + str(sys.argv[1]))
f = open(str(sys.argv[2]) ,"r")

X_Dom = []
Y_Dom = []

for line in f.readlines():
    A = line.split("   ")
    X_Dom.append(int(A[0]))
    Y_Dom.append((float(A[1].split("\n")[0])))

print(len(X_Dom))
print(len(Y_Dom))
plt.plot(X_Dom,Y_Dom)
plt.ylim([-400,0])
plt.savefig(str(sys.argv[3]))
