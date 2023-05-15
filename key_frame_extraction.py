import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import argrelextrema
from scipy.spatial.distance import pdist, squareform
from skimage.measure.entropy import shannon_entropy
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# %%

def concate(frame_index, entropy):
    concat_arr = []
    
    for i in range(0, len(frame_index)):
        concat_arr.append([frame_index[i], entropy[frame_index[i]]])
    return concat_arr

def compute_cutoff_distance(percent, N, data):
    # Input: 1. [percent]: If we choose 2.0 %, then the input should be 2.0 
    #        2. [N      ]: N clusters
    #        3. [data   ]: A 2-D array with at least 3 columns   
    position = round(N*percent/10); 
    data = np.asarray(data)
    data[:,2].sort()
    dc = data[:,2][position]
    return dc

def chi(x):
    if (x < 0):
        return 1
    else:
        return 0

def Pdist(arr, k, l):
    # Usage: The input array must be a 2-D array of [frame, entropy]
    return squareform(pdist(arr))[k][l]

def local_density(arr, k, dc):
    # Input: 1. [arr]: A 2-D array of [frame, entropy]
    #        2. [k  ]: Index of the current element of the 2-D array above
    #        3. [dc ]: A cut-off distance         
    
    rho = 0
    
    # Paper: Fast and Robust Dynamic Hand Gesture Recognition via
    #        Key Frames Extraction and Feature Fusion
    # Note : This calculation is derived from the formula (6)
    
    # for l in range(0, len(arr)):
    #     rho += chi(Pdist(arr, k, l) - dc)

    # Paper: Fast and Robust Dynamic Hand Gesture Recognition via
    #        Key Frames Extraction and Feature Fusion
    # Note : This calculation is derived from the formula (7)

    for l in range(0, len(arr)):
        if (l == k):
            continue
        else:
            rho += math.exp(-pow(Pdist(arr, k, l)/dc, 2))

    return rho

def min_frame_distance(arr_rho, arr_peak, k):
    # Input: 1. [arr_rho ]: An array of rho
    #        2. [arr_peak]: An array of local extrema frames
    #        3. [k       ]: Index of the current point
    
    if (len(arr_rho) != len(arr_peak)):
        raise ValueError("Two arrays must have the same length")
    
    sigma = 0
    
    for i in range(k, len(arr_peak)):
        for j in range(0, len(arr_peak)):
            if (arr_rho[j] > arr_rho[i]):
                sigma = abs(arr_peak[j] - arr_peak[i])
                break
        break
        
    return sigma

def cluster(arr_rho, arr_sigma, N):
    X = []
    
    for i in range(0, len(arr_rho)):
        X.append([arr_rho[i], arr_sigma[i]])
    
    
    kmeans = KMeans(n_clusters=N, random_state=0).fit(X)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    
    return kmeans.cluster_centers_, closest

# %%

def key_frame_extraction(file_dir, N, dc):
    # Input: 1. [file_dir]: The directory of the video file
    #        2. [N       ]: A pre-defined constant
    
    frame_entropy = []

    # Get frames from a video
    cap = cv2.VideoCapture(file_dir)
    
    # Count the framerate
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print("Frame rate: ", fps)
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        frame_entropy.append(shannon_entropy(frame))
    
    # Find local maximum points (P_max)
    p1 = list(argrelextrema(np.asarray(frame_entropy), np.greater)[0])

    # Find local minimum points (P_min)
    p2 = list(argrelextrema(np.asarray(frame_entropy), np.less)[0])

    # Union local maximum and local minimum points (P_extreme = union(P_max, P_min))
    local_peak = p1 + p2
    local_peak.sort()
    
    # Find local density     
    rho = []
    
    for i in range(0, len(concate(local_peak, frame_entropy))):
        rho.append(local_density(concate(local_peak, frame_entropy), i, dc))

    # Find minimum distance between frames
    sigma = []
    
    for i in range(0, len(rho)):
        sigma.append(min_frame_distance(rho, local_peak, i))

    # Clustering
    center, key_frame_index = cluster(rho, sigma, N)
    
    # Get key frames after clustering
    key_frame = []
    
    for i in range(0, len(key_frame_index)):
        key_frame.append(local_peak[key_frame_index[i]]) 
    
    key_frame.sort()
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Plot frame entropy
    plt.plot(np.asarray(frame_entropy))
    plt.grid(True)
    plt.show()
    
    # Plot the decision graph 
    plt.scatter(rho, sigma)
    plt.grid(True)
    plt.show()
    
    return key_frame

#%%

def key_frame_plot(file_dir, key_frame):
    
    # Get frames from a video
    cap = cv2.VideoCapture(file_dir)
    
    try: 
        os.mkdir('Frame')
    except:
        filelist = [ f for f in os.listdir('Frame') if f.endswith(".jpg") ]
        for f in filelist:
            os.remove(os.path.join('Frame', f))
    
    os.chdir(r'Frame')
    
    for i in range(0, len(key_frame)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, key_frame[i])
        ret, frame = cap.read()
        
        #cv2.imshow('Key frame ' + str(key_frame[i]), frame)
        cv2.imwrite('Key frame ' + str(key_frame[i]) + '.jpg', frame)
        cv2.waitKey(0)
        
    cap.release()