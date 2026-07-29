import cv2

img = cv2.imread("Week2/images/face1.jpg")

flipped = cv2.flip(img, 1)

rotated = cv2.rotate(
    img,
    cv2.ROTATE_90_CLOCKWISE
)

cv2.imwrite(
    "Week2/images/flipped_face1.jpg",
    flipped
)

cv2.imwrite(
    "Week2/images/rotated_face1.jpg",
    rotated
)

print("Data Augmentation Completed")