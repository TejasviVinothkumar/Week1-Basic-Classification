import cv2
import os

image_folder = "Week2/images"

for image_name in os.listdir(image_folder):

    if image_name.startswith("processed_"):
        continue

    if image_name.startswith("features_"):
        continue

    if image_name.startswith("flipped_"):
        continue

    if image_name.startswith("rotated_"):
        continue

    image_path = os.path.join(image_folder, image_name)

    img = cv2.imread(image_path)

    if img is None:
        continue

    resized = cv2.resize(img, (224, 224))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    output_path = os.path.join(
        image_folder,
        "processed_" + image_name
    )

    cv2.imwrite(output_path, gray)

    print("Processed:", image_name)

print("Dataset Preparation Completed")