"""
CS131 - Computer Vision: Foundations and Applications
Assignment 4
Author: Donsuk Lee (donlee90@stanford.edu)
Date created: 09/2017
Last modified: 10/9/2020
Python Version: 3.5+
"""

import numpy as np
import random
from scipy.spatial.distance import squareform, pdist, cdist
from skimage.util import img_as_float

### Clustering Methods
def kmeans(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    K-Means algorithm can be broken down into following steps:
        1. Randomly initialize cluster centers
        2. Assign each point to the closest center
        3. Compute new center of each cluster
        4. Stop if cluster assignments did not change
        5. Go to step 2

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        ### YOUR CODE HERE
        new_assignments = np.zeros_like(assignments)
        for feat_idx, feature in enumerate(features): 
            dist = float("inf")
            dist_idx = 0
            for center_idx, center in enumerate(centers): 
                temp_dist = np.linalg.norm(feature - center)
                if temp_dist < dist: 
                    dist = temp_dist
                    dist_idx = center_idx
            new_assignments[feat_idx] = dist_idx
        
        new_centers = np.zeros_like(centers)
        centers_count = np.zeros((new_centers.shape[0]))
        for assign_idx, assignment in enumerate(new_assignments): 
            new_centers[assignment] += features[assign_idx]
            centers_count[assignment] += 1
            
        for new_center_idx in range(len(new_centers)): 
            new_centers[new_center_idx] /= centers_count[new_center_idx]       
        if np.array_equal(new_assignments, assignments): 
            break
        centers = new_centers
        assignments = new_assignments
        ### END YOUR CODE

    return assignments

def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find cdist (imported from scipy.spatial.distance) and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Randomly initalize cluster centers
    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        ### YOUR CODE HERE
        dist_mat = cdist(features, centers)
        k_mat = np.argmin(dist_mat, axis=1)
        if np.array_equal(k_mat, assignments): 
            break
        
        new_centers = np.zeros_like(centers)
        centers_count = np.zeros((new_centers.shape[0]))
        for assign_idx, assignment in enumerate(k_mat): 
            new_centers[assignment] += features[assign_idx]
            centers_count[assignment] += 1
            
        for new_center_idx in range(len(new_centers)): 
            new_centers[new_center_idx] /= centers_count[new_center_idx]       
        centers = new_centers
        assignments = k_mat       
        ### END YOUR CODE

    return assignments

def hierarchical_clustering(features, k):
    """ Run the hierarchical agglomerative clustering algorithm.

    The algorithm is conceptually simple:

    Assign each point to its own cluster
    While the number of clusters is greater than k:
        Compute the distance between all pairs of clusters
        Merge the pair of clusters that are closest to each other

    We will use Euclidean distance to define distance between clusters.

    Recomputing the centroids of all clusters and the distances between all
    pairs of centroids at each step of the loop would be very slow. Thankfully
    most of the distances and centroids remain the same in successive
    iterations of the outer loop; therefore we can speed up the computation by
    only recomputing the centroid and distances for the new merged cluster.

    Even with this trick, this algorithm will consume a lot of memory and run
    very slowly when clustering large set of points. In practice, you probably
    do not want to use this algorithm to cluster more than 10,000 points.

    Hints
    - You may find pdist (imported from scipy.spatial.distance) useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """



    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    # Assign each point to its own cluster
    assignments = np.arange(N, dtype=np.uint32)
    centers = np.copy(features)
    n_clusters = N
    
    while n_clusters > k:
        ### YOUR CODE HERE
        # Computer distances between all pairs
        dists = pdist(centers)
        square = squareform(dists)
        np.fill_diagonal(square, float('inf'))

        # Find closest cluster pairs (i, j)  where j is the bigger index 
        square_idx = np.argmin(square)
        cluster_pair = list(np.unravel_index(square_idx, square.shape))
        cluster_pair = np.sort(cluster_pair)
        
        # Reassign cluster j's points to the small cluster i and delete index j
        for assign_idx, assignment in enumerate(assignments): 
            if assignment == cluster_pair[1]: 
                assignments[assign_idx] = cluster_pair[0]
            if assignment > cluster_pair[1]: 
                assignments[assign_idx] -= 1

        # Update centers: delete cluster j's center, recompute cluster i's center
        n_clusters -= 1
        new_center = np.zeros((centers.shape[1]))
        center_group_count = 0
        for assign_idx, assignment in enumerate(assignments): 
            if assignment == cluster_pair[0]: 
                new_center += features[assign_idx]
                center_group_count += 1

        new_center /= center_group_count
        centers[cluster_pair[0]] = new_center
        centers = np.delete(centers, cluster_pair[1], axis=0)
        ### END YOUR CODE

    return assignments


### Pixel-Level Features
def color_features(img):
    """ Represents a pixel by its color.

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))

    ### YOUR CODE HERE
    features = np.reshape(img, (H*W, C))
    ### END YOUR CODE

    return features

def color_position_features(img):
    """ Represents a pixel by its color and position.

    Combine pixel's RGB value and xy coordinates into a feature vector.
    i.e. for a pixel of color (r, g, b) located at position (x, y) in the
    image. its feature vector would be (r, g, b, x, y).

    Don't forget to normalize features.

    Hints
    - You may find np.mgrid and np.dstack useful
    - You may use np.mean and np.std

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C+2)
    """
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    ### YOUR CODE HERE

    res = np.zeros((H, W, C+2))
    for h in range(H): 
        for w in range(W): 
            for i in range(C): 
                res[h][w][i] = img[h][w][i]
            res[h][w][i + 1] = w
            res[h][w][i + 2] = h
            
    features = np.reshape(res, (H*W, C+2))
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    ### END YOUR CODE

    return features

def my_features(img):
    """ Implement your own features

    Args:
        img - array of shape (H, W, C)

    Returns:
        features - array of (H * W, C)
    """
    features = None
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE
    return features


### Quantitative Evaluation
def compute_accuracy(mask_gt, mask):
    """ Compute the pixel-wise accuracy of a foreground-background segmentation
        given a ground truth segmentation.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        mask - The estimated foreground-background segmentation. A logical
            array of the same size and format as mask_gt.

    Returns:
        accuracy - The fraction of pixels where mask_gt and mask agree. A
            bigger number is better, where 1.0 indicates a perfect segmentation.
    """

    accuracy = None
    ### YOUR CODE HERE
    tp = 0
    tn = 0
    pn = 0
    for i in range(len(mask_gt)):
        for j in range(len(mask_gt[1])):
            if mask_gt[i][j] == mask[i][j]:
                if mask_gt[i][j] == 1:
                    tp += 1 
                else: 
                    tn+=1
            pn += 1
    accuracy = (tp + tn) / pn
    ### END YOUR CODE

    return accuracy

def evaluate_segmentation(mask_gt, segments):
    """ Compare the estimated segmentation with the ground truth.

    Note that 'mask_gt' is a binary mask, while 'segments' contain k segments.
    This function compares each segment in 'segments' with the ground truth and
    outputs the accuracy of the best segment.

    Args:
        mask_gt - The ground truth foreground-background segmentation. A
            logical of size H x W where mask_gt[y, x] is 1 if and only if
            pixel (y, x) of the original image was part of the foreground.
        segments - An array of the same size as mask_gt. The value of a pixel
            indicates the segment it belongs.

    Returns:
        best_accuracy - Accuracy of the best performing segment.
            0 <= accuracy <= 1, where 1.0 indicates a perfect segmentation.
    """

    num_segments = np.max(segments) + 1
    best_accuracy = 0

    # Compare each segment in 'segments' with the ground truth
    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)

    return best_accuracy
