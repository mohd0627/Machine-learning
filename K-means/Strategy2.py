
# coding: utf-8

# In[1]:


from Precode2 import *
import numpy
import random
import matplotlib.pyplot as plt
data = np.load('AllSamples.npy')


# In[2]:


k1,i_point1,k2,i_point2 = initial_S2('6606') # please replace 0111 with your last four digit of your ID


# In[3]:


print(k1)
print(i_point1)
print(k2)
print(i_point2)


# In[94]:


def is_equal(a , b):
    for value in b:
        if a[0] == value[0] and a[1] == value[1]:
            return True
        
    return False
    
    
k = list(range(2,11))
output ={}
loss ={}
for x in k:
    output[x] = []
for i in k:
    #if i == k1:
    #    centroids = [i_point1]
    #elif i == k2:
    #    centroids = [i_point2]
    #else:
    random_index = random.randint(0, len(data)-1)
    centroids = [data[random_index,:]]
    dmax = 0
    for p1 in data:
        c1 = centroids[0]
        d1 = (p1[0]-c1[0])**2 + (p1[1]-c1[1])**2
        d1 = d1**0.5
        if d1>dmax:
            dmax = d1
            center = p1        
    centroids.append(center)
    if i>2:
        dist = numpy.zeros(300)
        for v in range(2,i):
            for p in data:
                d_avg=0
                for c in centroids:
                    d = (p[0]-c[0])**2 + (p[1]-c[1])**2
                    d = (d**0.5)/float(len(centroids))
                    d_avg = d_avg+d
                ind = numpy.where(data==p)
                dist[ind[0][0]] = d_avg
            dsorted = numpy.argsort(-1*dist)[:300]
            index=0
            flag = 1
            while flag == 1:
                if is_equal(data[index], centroids):
                    index+=1
                else:
                    centroids.append(data[index])
                    flag=0
                        
            
            
    for n in range(0,10):
        ds = numpy.empty([len(data), len(centroids)])
        for p in range(0,len(data)):
            p2d = data[p]
            for c in range(0,len(centroids)):
                center = centroids[c]
                d = (p2d[0]-center[0])**2 + (p2d[1]-center[1])**2
                d = d**0.5
                ds[p][c] = d
        d_min = numpy.argmin(ds, axis=1)
        clusters = {}
        for j in range(0,len(centroids)):
            clusters[j] = []

        for key in clusters.keys():
            inx = 0
            for index in range(0,len(data)):
                if d_min[index] == key:
                    clusters[key].insert(inx, data[index])
                    inx+=1
                    
        for key in clusters.keys():
            centroids[key] = (sum(clusters[key])/len(clusters[key]))
    s = 0
    for key in clusters.keys():
        points = clusters[key]
        center = centroids[key]
        for point in points:
            temp= (point[0]-center[0])**2 + (point[1]-center[1])**2
            s= s+temp
        
    loss[i] = s        
    output[i] = centroids
    

plt.scatter(loss.keys(), loss.values())
plt.xlabel('K (# of clusters)')
plt.ylabel('J (loss value)')

