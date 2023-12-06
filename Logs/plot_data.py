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
    X_Dom.append(int(A[0])*100)
    Y_Dom.append((float(A[1].split("\n")[0])))

print(len(X_Dom))
print(len(Y_Dom))
plt.plot(X_Dom,Y_Dom)
plt.plot(X_Dom,[-50]*len(X_Dom),'b')
plt.ylim([-200,0])
plt.xlim([5000,100000])
plt.xscale("log")
plt.xlabel("NEpoch")
plt.ylabel("AverageReward")
plt.title("[Evaluation] AverageReward in each Epoch")
plt.savefig(str(sys.argv[3]))
