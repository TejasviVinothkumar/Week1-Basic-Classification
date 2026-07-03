import tensorflow as tf

train_dir = "week3/dataset/train"
test_dir = "week3/dataset/test"

IMG_SIZE = (48, 48)
BATCH_SIZE = 32

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

print("\nDataset Loaded Successfully!")

print("\nEmotion Classes:")
print(train_dataset.class_names)