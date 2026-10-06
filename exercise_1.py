# Copyright 2026 Universidad Politécnica de Madrid
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import argparse
import os

import cv2
import numpy as np

Point = tuple[float, float]


class SegmentationLabel:

    def __init__(self, polygon: list[Point] | None = None, class_id: int | None = None):

        self.polygon: list[Point] | None = polygon
        self.class_id: int | None = class_id


# I know that the loop is slow and that the detection of blue
# could be better by transforming the image into HSV

def segment_image(image: np.ndarray) -> list[SegmentationLabel]:

    # TODO (Exercise 1): Implement an image segmentation algorithm
    # that takes an image as input and returns a list of SegmentationLabel objects.

    blue_pixels = []

    # Mask of black pixels; later, possible blue pixels will become white
    blue_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for y in range(image.shape[0]):

        for x in range(image.shape[1]):

            pixel = image[y, x]

            if pixel[0] > pixel[1] and pixel[0] > pixel[2]:

                blue_pixels.append((x, y))
                blue_mask[y, x] = 255

    contours, _ = cv2.findContours(
        blue_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
##i need to stop all the tiny countours beign countours
    segmentation = []
    
    for contour in contours:
        # Ignore small contours
        area = cv2.contourArea(contour)

        

        if area < 100:
            continue

        polygon = []

        for point in contour:

            x, y = point[0]

            polygon.append((x, y))

            segmentation.append(
                SegmentationLabel(
                    polygon=polygon,
                    class_id=0
        )
    )

    return segmentation
  
    #now we use the mask to find the gates loc
    ## we want the ocation of the whhite pixels, now them showing m the mask so, we cretae point a variable to store the coordinates
    #we need to create a polygon form the possible coordinates in orer to create segment objects
    ##only issue is that i dont take into account hat the gate is a frame idk how to solve this yet
    
gt_mask = None
prediction_mask = None
def compute_iou(
    image: np.ndarray, labels: list[SegmentationLabel], segmentation: list[SegmentationLabel]
) -> float:
    global gt_mask
    global prediction_mask
    
    height, width = image.shape[:2]

    # Create an empty mask for the ground truth
    gt_mask = np.zeros((height, width),dtype=np.uint8)

    # Create an empty mask for  prediction
    prediction_mask = np.zeros((height, width),dtype=np.uint8 )

    # Fill the ground-truth polygons
    for label in labels:
        if label.class_id == 0 and label.polygon is not None:

            polygon = []

            for point in label.polygon:
                x, y = point

                x = x * width
                y = y * height

                polygon.append((x, y))

        polygon = np.array(polygon, dtype=np.int32)

        cv2.fillPoly(gt_mask, [polygon], 1)
    # Fill  predicted polygons
    for label in segmentation:

        if label.class_id == 0 and label.polygon is not None:

            polygon = np.array(label.polygon,dtype=np.int32)

            cv2.fillPoly(prediction_mask,[polygon],1)
     
    
    intersection = np.logical_and(
       gt_mask,
        prediction_mask
    )

    union = np.logical_or(
        gt_mask,
        prediction_mask
    )
    intersection_pixels = np.sum(intersection)
    union_pixels = np.sum(union)

    if union_pixels == 0:
        return 0.0

    iou = intersection_pixels / union_pixels

    return float(iou)
    # TODO (Exercise 1): Implement a function that computes the Intersection over Union (IoU) score between the ground truth labels and the predicted segmentation.
    #were gonna fill in th etwo polygons, the ground truth one and the one made by my segmentation and then compare
    
    return iou


def compute_precision(
    image: np.ndarray, labels: list[SegmentationLabel], segmentation: list[SegmentationLabel]
) -> float:
    global gt_mask
    global prediction_mask

    TP = np.sum(
        np.logical_and(gt_mask, prediction_mask)
    )

    predicted_pixels = np.sum(prediction_mask)

    if predicted_pixels == 0:
        return 0.0

    return float(TP / predicted_pixels)

    # TODO (Exercise 1): Implement a function that computes the Precision score between the ground truth labels and the predicted segmentation.
    


def compute_recall(
    image: np.ndarray, labels: list[SegmentationLabel], segmentation: list[SegmentationLabel]
) -> float:
    global gt_mask
    global prediction_mask

    TP = np.sum(
        np.logical_and(gt_mask, prediction_mask)
    )

    actual_pixels = np.sum(gt_mask)

    if actual_pixels == 0:
        return 0.0

    return float(TP / actual_pixels)

    # TODO (Exercise 1): Implement a function that computes the Recall score between the ground truth labels and the predicted segmentation.
    


def load_dataset(dataset_path: str) -> list[tuple[cv2.Mat, list[SegmentationLabel]]]:
    dataset = []
    images_path = os.path.join(dataset_path, 'images')
    if not os.path.exists(images_path):
        print(f'Images directory not found: {images_path}')
        return dataset

    labels_path = os.path.join(dataset_path, 'labels')
    if not os.path.exists(labels_path):
        print(f'Labels directory not found: {labels_path}')
        return dataset

    for filename in os.listdir(images_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(images_path, filename)
            image: np.ndarray | None = cv2.imread(image_path)
            if image is None:
                print(f'Failed to load image: {image_path}')
                continue

            label_path = os.path.join(
                labels_path, filename.replace('.jpg', '.txt').replace('.png', '.txt')
            )
            labels: list[SegmentationLabel] = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    polygon = list(map(float, parts[1:]))
                    polygon = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
                    labels.append(SegmentationLabel(polygon=polygon, class_id=class_id))

            dataset.append((image, labels))
    return dataset


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Load a dataset of images and segmentation labels.'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        help='Path to the dataset directory',
        default='../datasets/yolo_masks',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    dataset = load_dataset(args.dataset_path)
    print(f'Loaded {len(dataset)} images and their corresponding labels.')

    result: list[list[SegmentationLabel]] = []
    for image, labels in dataset:
        segmentation = segment_image(image)
        result.append(segmentation)

    iou_scores = []
    recall_scores = []
    precision_scores = []
    for (image, labels), segmentation in zip(dataset, result):
        iou = compute_iou(image, labels, segmentation)
        iou_scores.append(iou)

        precision = compute_precision(image, labels, segmentation)
        precision_scores.append(precision)

        recall = compute_recall(image, labels, segmentation)
        recall_scores.append(recall)

    average_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0
    average_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
    average_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0

    print(f'Average IoU score: {average_iou:.4f}')
    print(f'Average Precision score: {average_precision:.4f}')
    print(f'Average Recall score: {average_recall:.4f}')
