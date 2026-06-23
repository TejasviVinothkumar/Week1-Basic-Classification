import numpy as np
import cv2
import os

os.makedirs("Week2/images", exist_ok=True)

for i in range(1, 4):
    img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    cv2.imwrite(f"Week2/images/face{i}.jpg", img)

print("3 sample images created successfully!")