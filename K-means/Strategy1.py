
# coding: utf-8

# In[2]:


from Precode import *
import numpy
import matplotlib.pyplot as plt
data = np.load('AllSamples.npy')


# In[3]:


k1,i_point1,k2,i_point2 = initial_S1('6606') # please replace 0111 with your last four digit of your ID


# In[4]:


print(k1)
print(i_point1)
print(k2)
print(i_point2)


# In[5]:


plt.scatter(data[:,0], data[:,1])
plt.show()


# In[6]:


k = list(range(2,11))
output ={}
loss ={}
for x in k:
    output[x] = []
for i in k:
    centroids = [[]]
    if i == k1:
        centroids = i_point1
    elif i == k2:
        centroids = i_point2
    else:
        random_indices = numpy.random.choice(len(data), size=i, replace=False)
        centroids = data[random_indices,:]
    for n in range(0,100):
        ds = numpy.empty([len(data), len(centroids)])
        for p in range(0,len(data)):
            p2d = data[p]
            for c in range(0,len(centroids)):
                center = centroids[c]
                d = (p2d[0]-center[0])**2 + (p2d[1]-center[1])**2
                d = d**0.5
                ds[p][c] = d
        d_min = numpy.argmin(ds, axis=1)
        clusters ={}
        for j in range(0,len(centroids)):
            clusters[j] = []
        for key in clusters.keys():
            for index in range(0,len(data)):
                if d_min[index] == key:
                    clusters[key].insert(index, data[index])
                    
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

