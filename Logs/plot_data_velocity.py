import os
import matplotlib.pyplot as plt
import sys
import numpy as np

#print(sys.argv[1])
os.chdir("./" + str(sys.argv[1]))
f = open(str(sys.argv[2]) ,"r")

T = [0.0473743566354, 2.195070206, 2.97991511967, 0.488632599809, 0.379904339073]
X_Dom = []
Y_Dom = []

X_Dom = []
Y_Dom = []

for line in f.readlines():
    A = line.split("   ")
    if int(A[1]) == int(sys.argv[4]) + 35:
        X_Dom.append(int(A[0]))
        Y_Dom.append(float(A[2]))

#print(len(X_Dom))
#print(len(Y_Dom))
plt.figure()
plt.plot(X_Dom,Y_Dom,'g')
plt.plot(X_Dom,[T[int(sys.argv[4])]]*len(X_Dom), 'r')
plt.title(str(sys.argv[4]))
#plt.ylim([-400, 0])
plt.savefig(str(sys.argv[3]))

print("Evaluation Tasks: ")
print("Task T", int(sys.argv[4])+35)

NUM_BUCKETS = 100
BUCKET_SIZE = 10
Buckets = [0]*NUM_BUCKETS

#print(len(X_Dom))
for i in range(1, len(X_Dom)+1):
    #print(i)
    if Y_Dom[i-1] > T[int(sys.argv[4])]:
        Buckets[int((i-1)/BUCKET_SIZE)] += 1

display_10 = False
display_25 = False
display_50 = False
display_70 = False
display_80 = False
display_90 = False
display_98 = False
display_100 = False

for i in range(NUM_BUCKETS):
    #print("Bucket - ", i, ": ", Buckets[i])
    if Buckets[i] >= 0.1*BUCKET_SIZE and not display_10:
        print("10p of values are above the Goal_Velocity at Iteration: ", (i+1)*BUCKET_SIZE*100)
        display_10 = True 
    if Buckets[i] >= 0.25*BUCKET_SIZE and not display_25:
        print("25p of values are above the Goal_Velocity at Iteration: ", (i+1)*BUCKET_SIZE*100)
        display_25 = True 
    if Buckets[i] >= 0.5*BUCKET_SIZE and not display_50:
        print("50p of values are above the Goal_Velocity at Iteration: ", (i+1)*BUCKET_SIZE*100)
        display_50 = True
    if Buckets[i] >= 0.7*BUCKET_SIZE and not display_70:
        print("70p of values are above the Goal_Velocity at Iteration: ", (i+1)*BUCKET_SIZE*100)
        display_70 = True
    if Buckets[i] >= 0.8*BUCKET_SIZE and not display_80:
        print("80p of values are above the Goal_Velocity at Iteration: ", (i+1)*BUCKET_SIZE*100)
        display_80 = True
    if Buckets[i] >= 0.9*BUCKET_SIZE and not display_90:
        print("90p of values are above the Goal_Velocity at Iteration: ", (i+1)*BUCKET_SIZE*100)
        display_90 = True
    if Buckets[i] >= 0.95*BUCKET_SIZE and not display_90:
        print("95p of values are above the Goal_Velocity at Iteration: ", (i+1)*BUCKET_SIZE*100)
        display_90 = True
    if Buckets[i] >= 0.98*BUCKET_SIZE and not display_98:
        print("98p of values are above the Goal_Velocity at Iteration: ", (i+1)*BUCKET_SIZE*100)
        display_98 = True
    if Buckets[i] >= 1*BUCKET_SIZE and not display_100:
        print("100p of values are above the Goal_Velocity at Iteration: ", (i+1)*BUCKET_SIZE*100)
        display_100 = True

print("Max number values: ", (np.max(np.array(Buckets))/BUCKET_SIZE)*100 ,"p that are above the Goal_Velocity at Iteration: ", (np.argmax(np.array(Buckets))+1)*BUCKET_SIZE*100)
print("Average of Last iterations around 100000 - percentage: ", (Buckets[NUM_BUCKETS-1]/BUCKET_SIZE)*100)


    
plt.figure()
plt.bar((np.arange(1,NUM_BUCKETS+1))*BUCKET_SIZE*100, np.array(Buckets)*1000, width=BUCKET_SIZE*80)
plt.title(str(sys.argv[4]))
plt.savefig("Bar_Graph_Efficiency_T" +str(sys.argv[4]) + ".png")

plt.figure()
plt.plot((np.arange(1,NUM_BUCKETS+1))*BUCKET_SIZE*100, np.array(Buckets)*1000)
plt.title(str(sys.argv[4]))
plt.savefig("Efficiency_T" +str(sys.argv[4]) + ".png")
    
    

