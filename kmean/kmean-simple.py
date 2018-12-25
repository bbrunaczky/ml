#!/usr/bin/python3

import pprint
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

print("\n" + "*" * 20, "1-dimension", "*" * 20, "\n")
centers = [20, 40, 80]

v = []
for i in range(0, 20):
    center_index = int((len(centers)-0.1) * np.random.rand(1)[0])
    v.append(centers[center_index] + (np.random.rand(1)[0]-0.5)*5)

print("input vector\n", v)

v = np.vstack(v)
v = np.float32(v)
print("floated input vector\n", v)

num_of_clusters = 3

comp, label, center = cv2.kmeans(v,
                                 num_of_clusters,
                                 None,
                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1),
                                 10,
                                 cv2.KMEANS_RANDOM_CENTERS)
print("comp\n", comp)
print("label\n", label)
print("center\n", center)
