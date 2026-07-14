import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

dataset_root = "Project Data"
img_size = (128, 128)
batch_size = 32
epochs = 10


def load_train_data(root_dir):
    image_paths = []
    labels = []

    food_train_path = os.path.join(root_dir, 'Food', 'Train')

    if os.path.exists(food_train_path):
        for category in os.listdir(food_train_path):
            category_path = os.path.join(food_train_path, category)

            if os.path.isdir(category_path):
                for img_name in os.listdir(category_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_paths.append(os.path.join(category_path, img_name))
                        labels.append(0)

    fruit_train_path = os.path.join(root_dir, 'Fruit', 'Train')

    if os.path.exists(fruit_train_path):
        for category in os.listdir(fruit_train_path):
            img_dir = os.path.join(fruit_train_path, category, 'Images')
            if os.path.isdir(img_dir):
                for img_name in os.listdir(img_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_paths.append(os.path.join(img_dir, img_name))
                        labels.append(1)

    return np.array(image_paths), np.array(labels)


def load_validation_data(root_dir):
    image_paths = []
    labels = []

    food_val = os.path.join(root_dir, 'Food', 'Validation')
    if os.path.exists(food_val):
        for category in os.listdir(food_val):
            category_path = os.path.join(food_val, category)
            if os.path.isdir(category_path):
                for img_name in os.listdir(category_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_paths.append(os.path.join(category_path, img_name))
                        labels.append(0)

    fruit_val = os.path.join(root_dir, 'Fruit', 'Validation')
    if os.path.exists(fruit_val):
        for category in os.listdir(fruit_val):
            img_dir = os.path.join(fruit_val, category, 'Images')
            if os.path.isdir(img_dir):
                for img_name in os.listdir(img_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_paths.append(os.path.join(img_dir, img_name))
                        labels.append(1)

    return np.array(image_paths), np.array(labels)


def preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img


def create_dataset(paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(lambda p, l: (preprocess_image(p), l), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset



if __name__ == "__main__":

    train_paths, train_labels = load_train_data(dataset_root)
    print(f"Found {len(train_paths)} training images.")

    val_paths, val_labels = load_validation_data(dataset_root)
    print(f"Found {len(val_paths)} validation images.")

    train_ds = create_dataset(train_paths, train_labels)
    val_ds = create_dataset(val_paths, val_labels)

    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=img_size + (3,))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

    print("Starting training...")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    model.save("model_part_a_binary.h5")
    print("Model saved.")
