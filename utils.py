import cv2
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt
from skimage import transform as tf
from skimage.measure import label

import torch
from torchvision.ops import masks_to_boxes

def remove_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobel_mag = np.sqrt(sum([filters.sobel(gray, axis=i)**2
                             for i in range(gray.ndim)]) / gray.ndim)
    bin_sobel_mag  = sobel_mag > 0.1

    kernel1 = np.ones((3, 3), 'uint8')
    kernel2 = np.ones((7, 7), 'uint8')

    erode_img = bin_sobel_mag.astype('uint8')
    dilate_img = cv2.dilate(erode_img, kernel2, iterations=1)

    contours, hierarchy = cv2.findContours(dilate_img.astype('uint8'), cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(cnt) for cnt in contours]

    index = np.argmax(areas)

    rect = cv2.minAreaRect(contours[index])

    box = cv2.boxPoints(rect)
    box = np.int0(box)
    distances = [np.linalg.norm(point) for point in box]
    if np.argmin(distances) == 1:
        box = box[[1, 2, 3, 0]]
        
    w = np.sqrt(np.sum((box[1] - box[0])**2))
    h = np.sqrt(np.sum((box[2] - box[1])**2))

    w = int(w)
    h = int(h)

    src = np.array([[0, 0], [0, w], [h, w], [h, 0]])
    tform3 = tf.ProjectiveTransform()
    tform3.estimate(src, box)
    result_table = tf.warp(img, tform3, output_shape=(w, h))

    return result_table


def filter_boxes(boxes, len_points, th = 0.1):
    final_boxes = []
    
    boxes = np.array(boxes)
    x0 = boxes[:, 1]
    y0 = boxes[:, 0]
    x1 = boxes[:, 3]
    y1 = boxes[:, 2]
    ars = (x1 - x0) * (y1 - y0)

    ids = np.argsort(len_points)
    
    while len(ids) > 0:

        final_boxes.append(boxes[ids[-1]])

        x_1 = np.maximum(x0[ids[-1]], x0[ids[:-1]])
        x_2 = np.minimum(x1[ids[-1]], x1[ids[:-1]])
        y_1 = np.maximum(y0[ids[-1]], y0[ids[:-1]])
        y_2 = np.minimum(y1[ids[-1]], y1[ids[:-1]])

        inter = np.maximum(0.0, x_2 - x_1 + 1) * np.maximum(0.0, y_2 - y_1 + 1)
        iou = inter / (ars[ids[-1]] + ars[ids[:-1]] - inter)
        ids = ids[np.where(iou < th)]

    return final_boxes


def get_box_of_true_pill(new):
    new *= 255
    gray = cv2.cvtColor(new.astype('uint8'), cv2.COLOR_BGR2GRAY)
    sobel_mag = np.sqrt(sum([filters.sobel(gray, axis=i)**2
                                 for i in range(gray.ndim)]) / gray.ndim)
    bin_sobel_mag  = sobel_mag > 0.1
    kernel1 = np.ones((3, 3), 'uint8')
    kernel2 = np.ones((5, 5), 'uint8')
    dilate_img = cv2.dilate(bin_sobel_mag.astype('uint8'), kernel2, iterations=1)


    labeled_mask, count = label(1-dilate_img, return_num=True, connectivity = 2)
    areas = []
    ps = []
    for l in range(1, count + 1):
        component_mask = labeled_mask == l
        areas.append(np.sum(component_mask))

    indexes = np.argsort(areas)[-3:]
#     print(areas)
    
    component_mask = labeled_mask == indexes[0] + 1
    component_mask += labeled_mask == indexes[1] + 1
    component_mask += labeled_mask == indexes[2] + 1
#     plt.imshow(component_mask, 'gray')
#     component_mask += labeled_mask == indexes[3] + 1
#     component_mask += labeled_mask == indexes[4] + 1
    contours, hierarchy = cv2.findContours(component_mask.astype('uint8'), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    ps = [cv2.arcLength(cnt, True) for cnt in contours]

    component_mask = labeled_mask == indexes[np.argmin(ps)] + 1
    #plt.imshow(component_mask, 'gray')
    
    #return mask2bounding_box(component_mask)
    return masks_to_boxes(torch.from_numpy(component_mask).unsqueeze(0)).numpy().astype('int')

