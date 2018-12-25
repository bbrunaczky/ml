#!/usr/bin/python3

import pprint
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

box = [(20, 20), (80, 60)]
num_of_samples = 400
num_of_clusters = 3
colors = ["r", "g", "b", "c", "m", "y"]
markers = ["o", "v", "X"]


fig, axs = plt.subplots(1,2)
for ax in axs:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)


v = []
points = []
for i in range(0, num_of_samples):
    x = box[0][0] + np.random.rand(1)[0]*(box[1][0]-box[0][0]) 
    y = box[0][1] + np.random.rand(1)[0]*(box[1][1]-box[0][1])
    color = 0
    if x>20 and x<=40:
        color = 0
    elif x>40 and x<=60 and y>20 and y<=40:
        color = 1
    elif x>50 and x<=60 and y>20 and y<=50:
        color = 1
    else:
        color = 2
    
    # need to be normalized
    tup = (x/80., y/60., color/2.)
    v.append(tup)
    tup = (x, y, color)
    points.append(tup)
    axs[0].scatter(x, y, marker=markers[color % len(markers)], c=colors[color % len(colors)])

print("input vector\n", v)


v = np.vstack(v)
v = np.float32(v)
print("floated input vector\n", v)


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
    segments[index].append( (points[i][0], points[i][1]) )
    


for i in range(0,len(segments)):
    segment = segments[i]
    segment = np.vstack(segment)
    segment = np.float32(segment)
    print("segment", segment)
    x = segment[:,0]
    y = segment[:,1]
    axs[1].scatter(x, y, marker=markers[i%len(markers)], c=colors[i])

plt.show()
