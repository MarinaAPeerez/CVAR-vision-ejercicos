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

CornerLabel = tuple[float, float]


def order_quad(pts):
    """Order 4 points clockwise, starting from the top-left-most one."""
    pts = np.array(pts, dtype=np.float32)
    c = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(angles)]
    start = np.argmin(pts[:, 0] + pts[:, 1])  # top-left-most point
    pts = np.roll(pts, -start, axis=0)
    return [(float(p[0]), float(p[1])) for p in pts]


def sort_corners(points):
    """Return 4 outer corners (ordered) followed by 4 inner corners (ordered).

    Outer vs inner is decided by distance to the centroid of all 8 points,
    so it works for both the detections and the label files.
    """
    if len(points) != 8:
        return sorted(points, key=lambda p: (p[1], p[0]))
    pts = np.array(points, dtype=np.float32)
    c = pts.mean(axis=0)
    dist = np.linalg.norm(pts - c, axis=1)
    idx = np.argsort(-dist)  # farthest first
    outer = pts[idx[:4]]
    inner = pts[idx[4:]]
    return order_quad(outer) + order_quad(inner)


def detect_corners(image: np.ndarray) -> list[CornerLabel]:
    # 1-2. Blue mask (vectorized: blue channel greater than green and red)
    b, g, r = image[..., 0], image[..., 1], image[..., 2]
    blue_mask = ((b > g) & (b > r)).astype(np.uint8) * 255

    # 3. Boundaries of the blue regions
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 4. Keep only reasonably large regions
    polygons = [c for c in contours if cv2.contourArea(c) >= 100]
    if len(polygons) == 0:
        return []

    # Largest blue region
    contour = max(polygons, key=cv2.contourArea)

    # 5. Approximate with a polygon
    perimeter = cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    if len(polygon) != 4:
        return []

    # 6. Outer corners
    outer = [(float(p[0][0]), float(p[0][1])) for p in polygon]

    # 7-9. Keep only the area inside the outer polygon
    poly_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(poly_mask, [polygon], 255)
    inside_image = cv2.bitwise_and(image, image, mask=poly_mask)

    # 10. White mask (vectorized)
    white_mask = (inside_image > 150).all(axis=2).astype(np.uint8) * 255

    # Close the gaps left by the blue bars
    kernel = np.ones((15, 15), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    # 11. Boundaries of the white regions
    contours1, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 12. Keep large white regions
    polygons1 = [c for c in contours1 if cv2.contourArea(c) >= 100]
    if len(polygons1) == 0:
        return []

    contour1 = max(polygons1, key=cv2.contourArea)
    perimeter1 = cv2.arcLength(contour1, True)
    polygon1 = cv2.approxPolyDP(contour1, 0.02 * perimeter1, True)
    if len(polygon1) != 4:
        return []

    # 14. Inner corners
    inner = [(float(p[0][0]), float(p[0][1])) for p in polygon1]

    return outer + inner


def load_dataset(dataset_path: str) -> list[tuple[np.ndarray, list[CornerLabel]]]:
    dataset = []
    images_path = os.path.join(dataset_path, 'images')
    if not os.path.exists(images_path):
        print(f'Images directory not found: {images_path}')
        return dataset

    labels_path = os.path.join(dataset_path, 'labels')
    if not os.path.exists(labels_path):
        print(f'Labels directory not found: {labels_path}')
        return dataset

    for filename in sorted(os.listdir(images_path)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(images_path, filename)
            image: np.ndarray | None = cv2.imread(image_path)
            if image is None:
                print(f'Failed to load image: {image_path}')
                continue
            h, w = image.shape[:2]

            label_path = os.path.join(
                labels_path, filename.replace('.jpg', '.txt').replace('.png', '.txt')
            )
            if not os.path.exists(label_path):
                print(f'Label file not found: {label_path}')
                continue

            labels: list[CornerLabel] = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    parts = parts[5:]  # skip class cx cy w h
                    for i in range(0, len(parts) - 2, 3):
                        # FIX: labels are normalized [0, 1] -> convert to pixels
                        x = float(parts[i]) * w
                        y = float(parts[i + 1]) * h
                        labels.append((x, y))

            labels = sort_corners(labels)

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
        default='../datasets/yolo_pose',
    )
    return parser.parse_args()


def quad_error(c4, l4):
    """Mean error of a quad, plus the best error over the 4 cyclic shifts."""
    c4 = np.array(c4)
    l4 = np.array(l4)
    raw = np.linalg.norm(c4 - l4, axis=1).mean()
    best = min(
        np.linalg.norm(np.roll(c4, s, axis=0) - l4, axis=1).mean() for s in range(4)
    )
    return raw, best


if __name__ == '__main__':
    args = parse_arguments()
    dataset = load_dataset(args.dataset_path)
    print(f'Loaded {len(dataset)} images and their corresponding labels.')
    result: list[list[CornerLabel]] = []
    for image, labels in dataset:
        corners = detect_corners(image)
        corners = sort_corners(corners)
        result.append(corners)
    distances = [1]
    for corners, labels in zip(result, dataset):
        for corner, label in zip(corners, labels[1]):
            distance = np.sqrt((corner[0] - label[0]) ** 2 + (corner[1] - label[1]) ** 2)
            distances.append(distance)
    print(f'Average distance: {np.mean(distances)}')








    

        