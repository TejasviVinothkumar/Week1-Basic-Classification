import cv2
import os

image_path = "Week2/images/processed_face1.jpg"

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(img, 100, 200)

output_path = "Week2/images/features_face1.jpg"

cv2.imwrite(output_path, edges)

print("Feature Extraction Completed")