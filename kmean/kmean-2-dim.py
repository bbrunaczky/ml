#!/usr/bin/python3

import pprint
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

centers = [(20, 40), (40, 40), (80, 40)]
radius = 10
num_of_samples = 40
print("centers\n", centers)

v = []
for i in range(0, num_of_samples):
    center_index = int((len(centers)-0.1) * np.random.rand(1)[0])
    center = centers[center_index]
    x = center[0] + (np.random.rand(1)[0]-0.5)*2*radius 
    y = center[1] + (np.random.rand(1)[0]-0.5)*2*radius
    tup = (x, y)
    v.append(tup)

print("input vector\n", v)

v = np.vstack(v)
v = np.float32(v)
print("floated input vector\n", v)

num_of_clusters = 3
colors = ["r", "g", "b", "c", "m", "y"]
markers = ["o", "v", "X"]

comp, label, center = cv2.kmeans(v,
                                 num_of_clusters,
                                 None,
                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1),
                                 10,
                                 cv2.KMEANS_RANDOM_CENTERS)
print("comp\n", comp)
print("label\n", label)
print("center\n", center)


segments = [] 
for i in range(0, num_of_clusters):
    segments.append([])
    
for i in range(0,len(v)):
    index = label[i][0]
    print(index, v[i])
    segments[index].append( (v[i][0], v[i][1]) )
    
#print(segments)

fig = plt.figure()
ax = fig.add_subplot(111)

for i in range(0,len(segments)):
    segment = segments[i]
    segment = np.vstack(segment)
    segment = np.float32(segment)
    print("segment", segment)
    x = segment[:,0]
    y = segment[:,1]
    ax.scatter(x, y, marker=markers[i%len(markers)], c=colors[i])

plt.show()
