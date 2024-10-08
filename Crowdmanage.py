import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity

def km_seg(img, n_clusters=2):
    flattened_image = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(flattened_image)
    segmented_image = kmeans.cluster_centers_[kmeans.labels_]
    segmented_image = segmented_image.reshape(img.shape).astype(np.uint8)
    return segmented_image

def kde_den_pred(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_size = 15
    density_map = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
    return density_map

def color_code_den(img, density_map, thresholds=(50, 100)):
    dense_mask = density_map > thresholds[1]
    medium_dense_mask = (density_map <= thresholds[1]) & (density_map > thresholds[0])
    low_dense_mask = density_map <= thresholds[0]


    dense_color = np.array([0, 200, 0])
    medium_dense_color = np.array([0, 165, 255])
    low_dense_color = np.array([0, 0, 200])

    color_coded_image = np.zeros_like(img)
    color_coded_image[dense_mask] = dense_color
    color_coded_image[medium_dense_mask] = medium_dense_color
    color_coded_image[low_dense_mask] = low_dense_color

    return color_coded_image

def main(image_path):
    img = cv2.imread(image_path)

    seg_image = km_seg(img)

    den_map = kde_den_pred(img)

    color_coded_img = color_code_den(img, den_map)

    cv2.imshow('Original Image', img)
    cv2.imshow('Segmented Image', seg_image)
    cv2.imshow('Density Map', den_map)
    cv2.imshow('Color Coded Density', color_coded_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = r"C:\Users\singh\Downloads\c.jpg"
    main(image_path)