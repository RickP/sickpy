#!/usr/bin/python
import cv2
import numpy as np
import pandas as pd

from sick import *
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from geopy.distance import great_circle
from shapely.geometry import MultiPoint

MAX_DIST = 400 # cm !
MAX_X = 150 # cm !
MAX_Y = 250 # cm !

def make_image(cartesian):
    img = np.zeros(((MAX_DIST + 50),2*(MAX_DIST + 50),3), np.uint8)
    for c in cartesian:
        if c[2] > MAX_DIST: 
            continue
        cv2.circle(img,(int(c[0] + img.shape[1]/2 ),int(img.shape[0] - 20 - c[1])),2,(0,255*c[2]/MAX_DIST,255-255*c[2]/MAX_DIST),1)
    return img
    
def make_image2(cartesian):
    img = np.zeros(((MAX_DIST + 50),2*(MAX_DIST + 50),3), np.uint8)
    for c in cartesian:
        cv2.circle(img,(int(c[0] + img.shape[1]/2 ),int(img.shape[0] - 20 - c[1])),5,(0,255,128),1)
    return img
 
def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return centermost_point

print "<<<< initing sick"
sick1 = SICK("/dev/tty.usbserial-A600A9QG")
while True:
    if sick1.get_frame() and sick1.cartesian != None:
        
        points = []
        centermost_points =[]
        for point in sick1.cartesian:
            if point[0] >= -MAX_X and point[0] <= MAX_X and point[1] >= -MAX_Y and point[1] <= MAX_Y: 
                points.append([point[0], point[1]])
                
        if len(points) > 2:
            points = np.array(points)
            db = DBSCAN(eps=15, min_samples=3).fit(points)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            cluster_labels = db.labels_

            print(cluster_labels)

            # Number of clusters in labels, ignoring noise if present.
            num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            clusters = pd.Series([points[cluster_labels == n] for n in range(num_clusters)])
            
            
            for cluster in clusters:
                if len(cluster) < 25: centermost_points.append(get_centermost_point(cluster))
           
        if len(centermost_points) > 0:        
            cv2.imshow("make_image", make_image2(centermost_points))
        else:
            cv2.imshow("make_image", np.zeros(((MAX_DIST + 50),2*(MAX_DIST + 50),3), np.uint8))
            
        cv2.waitKey(1)
