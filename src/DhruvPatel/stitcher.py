import cv2
import numpy as np
from glob import glob
from natsort import natsorted
from tqdm import tqdm
import os
import sys

class Panaroma():
    
    def __init__(self):
        pass
    
    def sift_detector(self, img1, img2,MIN_MATCH_COUNT = 10, FLANN_INDEX_KDTREE = 1 ):
        sift = cv2.SIFT_create()

        # find the key points and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.85 *n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:

            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,2)
        else:
            print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
            matchesMask = None

        return src_pts, dst_pts



    def ransac(self, src_pts: int, dst_pts: int,m,n, corres_pts : int = 8, itrs = 10000):
        max_pts = src_pts.shape[0]
        t = 500

        # Checking for minimum point correspondences i.e 4.
        assert max_pts == dst_pts.shape[0], "Invalid Point Correspondences"
        assert max_pts > 4, "Minimum 4 point correspondences required"
        
        # Seperating the x and y coordinates
        src_x = src_pts[:,0]
        src_y = src_pts[:,1]
        dst_x = dst_pts[:,0]
        dst_y = dst_pts[:,1]

        opt_pts = []

        # Running for k times
        for i in tqdm(range(itrs), ncols = 100, desc = "running ransac: "):

            # Random Sampling 
            samples = np.random.randint(0, max_pts, corres_pts)

            # Equations of the point correspondences
            A = np.zeros((2*corres_pts,9), dtype=np.float32)

            T = [[2/n, 0, -n/2],
            [0, 2/m, -m/2],
            [0, 0, 1]]

            T = np.array(T, dtype = np.float32)

            for i,idx in enumerate(samples):
                x1 = np.array((src_x[idx],src_y[idx],1))
                x2 = np.array((dst_x[idx],dst_y[idx],1))

                n_x1 = T @ x1
                n_x2 = T @ x2

                A[2*i,3:] = [-n_x1[0], -n_x1[1], -1, n_x2[1]*n_x1[0], n_x2[1]*n_x1[1], n_x2[1]]
                A[2*i+1,:3] = [n_x1[0], n_x1[1], 1]
                A[2*i+1,-3:] = [ -n_x2[0]*n_x1[0], -n_x2[0]*n_x1[1], -n_x2[0]]
            
            _,_,V = np.linalg.svd(A)
            
            # Homography matrix --> (3,3)
            H = V[-1].reshape(3,3)

            H = np.linalg.inv(T) @ H @ T

            loss = 0.0
            dist = []
            # Calculating the inliears and taking point correspondences with max inliers.
            for i in range(max_pts):
                x = np.array((src_x[i],src_y[i],1), dtype = np.float32)
                y = np.array((dst_x[i],dst_y[i],1), dtype = np.float32)

                
                y_hat  = np.matmul(H,  x)

                loss = np.linalg.norm(y - y_hat)
                dist.append(loss)

            dist = np.array(dist, dtype = np.float32)

            index = np.where(dist < t)
            if(len(index) > len(opt_pts)):
                opt_pts = index[0]

        # Optimal src and dst coordinates.
        opt_src_x = src_x[opt_pts]
        opt_src_y = src_y[opt_pts]
        opt_dst_x = dst_x[opt_pts]
        opt_dst_y = dst_y[opt_pts]

        return opt_src_x, opt_src_y, opt_dst_x, opt_dst_y

    def dlt(self,src_pts, dst_pts, corres_pts,m,n):
        src_x = src_pts[:,0]
        src_y = src_pts[:,1]
        dst_x = dst_pts[:,0]
        dst_y = dst_pts[:,1]

        corres_pts = len(src_x)

        # Creating matrix T for normalization
        T = [[2/n, 0, -n/2],
            [0, 2/m, -m/2],
            [0, 0, 1]]
        T = np.array(T, dtype = np.float32)
        nsrc_x = []
        nsrc_y = []
        nsrc_w = []
        ndst_x = []
        ndst_y = []
        ndst_w = []

        # Calulating the normalized coordinates
        for i in range(corres_pts):
            x1 = np.array((src_x[i],src_y[i],1), dtype = np.float32)
            x2 = np.array((dst_x[i],dst_y[i],1), dtype = np.float32)

            n_x1 = np.matmul(T,x1)
            n_x2 = np.matmul(T,x2)
        
            nsrc_x.append(n_x1[0])
            nsrc_y.append(n_x1[1])
            nsrc_w.append(n_x1[2])
            ndst_x.append(n_x2[0])
            ndst_y.append(n_x2[1])
            ndst_w.append(n_x2[2])

        
        A = np.zeros((2*corres_pts,9), dtype=np.float32)

        # Estimating the normalized Homography matrix.
        for i in range(corres_pts):
            A[2*i,3:] = [-ndst_w[i]*nsrc_x[i], -ndst_w[i]*nsrc_y[i], -ndst_w[i], ndst_y[i]*nsrc_x[i], ndst_y[i]*nsrc_y[i], ndst_y[i]]
            A[2*i+1,:3] = [ndst_w[i]*nsrc_x[i], ndst_w[i]*nsrc_y[i], ndst_w[i]]
            A[2*i+1,-3:] = [-ndst_x[i]*nsrc_x[i], -ndst_x[i]*nsrc_y[i], -ndst_x[i]]
            
        _,_,V = np.linalg.svd(A)

        normalized_H = V[:,-1].reshape(3,3)


        T_inverse = np.linalg.pinv(T)

        # Estimating the un_normalized Homography matrix.
        H = T_inverse @ normalized_H @ T
        H = np.array(H, dtype = np.uint8)

        return H
    

    def homography_matrix(self,img1, img2, src_x, src_y, dst_x, dst_y):
        m,n,_ = img1.shape
        corres_pts = len(src_x)

        # Creating matrix T for normalization
        T = [[2/n, 0, -n/2],
            [0, 2/m, -m/2],
            [0, 0, 1]]
        T = np.array(T, dtype = np.float32)

        nsrc_x = []
        nsrc_y = []
        ndst_x = []
        ndst_y = []

        # Calulating the normalized coordinates
        for i in range(corres_pts):
            x1 = np.array((src_x[i],src_y[i],1), dtype = np.float32)
            x2 = np.array((dst_x[i],dst_y[i],1), dtype = np.float32)

            n_x1 = T @ x1
            n_x2 = T @ x2

            nsrc_x.append(n_x1[0])
            nsrc_y.append(n_x1[1])
            ndst_x.append(n_x2[0])
            ndst_y.append(n_x2[1])

        A = np.zeros((2*corres_pts,9), dtype=np.float32)

        # Estimating the normalized Homography matrix.
        for i in range(corres_pts):
            A[2*i,3:] = [-nsrc_x[i], -nsrc_y[i], -1, ndst_y[i]*nsrc_x[i], ndst_y[i]*nsrc_y[i], ndst_y[i]]
            A[2*i+1,:3] = [nsrc_x[i], nsrc_y[i], 1]
            A[2*i+1,-3:] = [-ndst_x[i]*nsrc_x[i], -ndst_x[i]*nsrc_y[i], -ndst_x[i]]
            
        _,_,V = np.linalg.svd(A)

        normalized_H = V[-1].reshape(3,3)


        T_inverse = np.linalg.inv(T)

        # Estimating the unnormalized Homography matrix.
        H = T_inverse @ normalized_H @ T
        H = np.array(H, dtype = np.uint8)
        return H
    
    def get_bounding_box(self, img1, img2, H):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        corners_img1 = np.array([[0, 0, 1], [w1, 0, 1], [0, h1, 1], [w1, h1, 1]]).T
        corners_img2 = np.array([[0, 0, 1], [w2, 0, 1], [0, h2, 1], [w2, h2, 1]]).T

        transformed_corners_img1 = H @ corners_img1
        transformed_corners_img1 = transformed_corners_img1.astype(float)
        transformed_corners_img1 /= transformed_corners_img1[2] # Normalize by third coordinate


        all_corners = np.hstack((transformed_corners_img1[:2], corners_img2[:2]))


        min_x, min_y = np.floor(np.min(all_corners, axis=1)).astype(int)
        max_x, max_y = np.ceil(np.max(all_corners, axis=1)).astype(int)

        width = max_x - min_x
        height = max_y - min_y

        return height, width, min_x, min_y

    def warp_image(self,image, H, output_shape, min_x, min_y):

        H_inv = np.linalg.inv(H)

        warped_image = np.zeros(output_shape + (image.shape[2],), dtype= np.float32)

        for y in range(output_shape[0]):
            for x in range(output_shape[1]):
                src_coords = H_inv @ [x + min_x, y + min_y, 1]
                src_coords /= src_coords[2] 
                
                src_x, src_y = int(src_coords[0]), int(src_coords[1])

                if 0 <= src_x < image.shape[1] and 0 <= src_y < image.shape[0]:
                    warped_image[y, x] = image[src_y, src_x]
                    
        return warped_image


    def blend_images(self, img1, img2, x_offset, y_offset):
        for x in range(img2.shape[0]):
            for y in range(img2.shape[1]):
                if np.any(img2[x, y] > 0):  
                    img1[x + x_offset, y + y_offset] = img2[x, y]
        
        return img1
    

    def create_panaroma(self, path):

        # Fetching All Images 
        gt_images = []
        for files in tqdm(natsorted(glob(path + '/*')),ncols= 100, desc = f'fetching images from {path.split(os.sep)[-1]}'):
            gt_images.append(cv2.imread(files, 1))
    
        gt_images = np.array(gt_images)

        length = len(gt_images)
        
        stiched_img = gt_images[0]
        homography_matrix = []

        # Calculating Homography matrix and stitching all images.
        for i in range(1,length):
            img1 = gt_images[i-1]
            img2 = gt_images[i]
            # fetches matches
            src_pts, dst_pts = tqdm(self.sift_detector(img1, img2), ncols = 100,  desc = "running SIFT: ")

            opt_src_x,opt_src_y, opt_dst_x, opt_dst_y = self.ransac(src_pts, dst_pts, img1.shape[0], img2.shape[1], 10, 1000)

            H = self.homography_matrix(img1, img2, opt_src_x, opt_src_y, opt_dst_x, opt_dst_y)
            # H = self.dlt(src_pts, dst_pts,src_pts.shape[0],img1.shape[0], img1.shape[1])

            if H[0,0] == 0:
                H[0,0] = 1
            if H[1,1] == 0:
                H[1,1] = 1
            if H[2,2] == 0:
                H[2,2] = 1

            homography_matrix.append(H)

            height, width, min_x, min_y = self.get_bounding_box(stiched_img, img2, H)
            output_shape = (height, width)
            warped_img = self.warp_image(img2, H, output_shape, min_x, min_y)
            stiched_img = self.blend_images(stiched_img, warped_img, min_x, min_y)
            
        return stiched_img, homography_matrix


