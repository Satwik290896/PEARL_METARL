import os
import matplotlib.pyplot as plt

f = open("Average_Reward_OfflineData_CheetahVel_R2.txt","r")
X_Dom = []
Y_Dom = []

for line in f.readlines():
    A = line.split("   ")
    X_Dom.append(int(A[0]))
    Y_Dom.append((float(A[1].split("\n")[0])))

print(len(X_Dom))
print(len(Y_Dom))
plt.plot(X_Dom,Y_Dom)
plt.savefig('Graph_R2.png')
