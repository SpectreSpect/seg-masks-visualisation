import ultralytics
import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_masks(path):
    masks = []
    with open(path, "r") as file:
        for line_idx, line in enumerate(file.readlines()):
            split_line = line.split(' ')
            category = split_line[0]
            if split_line[-1] == '\n':
                coords = split_line[1:-1]
            else:
                coords = split_line[1:]
            

            if len(coords) % 2 != 0:
                print(f"Invalid data - line_idx: {line_idx}  category: {category}  len(coords): {len(coords)}")
                return None
            
            nxy = []
            for idx in range(0, len(coords), 2):
                nx = float(coords[idx])
                ny = float(coords[idx + 1])
                nxy.append([nx, ny])
            
            mask = {"nxy": np.array(nxy), "category": category}
            masks.append(mask)
    return masks


def draw_segmentation_cv2(image, masks, palette, alpha=0.3):
    if masks is None:
        return []

    mask_image = np.zeros(image.shape[:-1], dtype=np.uint8)
    for idx, mask in enumerate(masks):
        color = palette[idx % len(palette)]

        nxy = mask["nxy"]
        nxy[:, 0] *= image.shape[1]
        nxy[:, 1] *= image.shape[0]

        cv2.drawContours(mask_image, [np.expand_dims(nxy, 1).astype(int)], contourIdx=-1, color=(255), thickness=-1)
        
        indices = mask_image != 0 
        image[indices] = image[indices] * (1 - alpha) + np.array(color) * alpha
        mask_image[:] = 0

    return image
            


if __name__ == "__main__":
    masks = get_masks("data/yolov8_medium-1000_2/labels/train/155320867521191500.txt")
    image = cv2.imread("data/yolov8_medium-1000_2/images/train/155320867521191500.jpg")

    small_image = cv2.resize(image, None, fx=0.5, fy=0.5)

    seg_image = draw_segmentation_cv2(small_image, masks, [(255, 0, 255), (0, 255, 255), (255, 255, 0)])

    cv2.imshow('Image', seg_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()