import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam



train_dir = "Project Data/Fruit/Train"
val_dir = "Project Data/Fruit/Validation"
img_size = (128, 128)
batch_size = 32
epochs = 15


# --- 1. Data Loading & Label Encoding ---

def get_class_names(root_dir):
    return sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])


def load_data(root_dir, class_to_index):

    filepaths = []
    labels = []

    if not os.path.exists(root_dir):
        print(f"Warning: {root_dir} does not exist.")
        return np.array([]), np.array([])

    for fruit_class in class_to_index.keys():
        # Handle 'Images' subfolder structure for Fruit
        class_path = os.path.join(root_dir, fruit_class)
        img_dir = os.path.join(class_path, 'Images')

        # Fallback if 'Images' folder doesn't exist
        if not os.path.isdir(img_dir):
            img_dir = class_path

        if os.path.isdir(img_dir):
            for file in os.listdir(img_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepaths.append(os.path.join(img_dir, file))
                    labels.append(class_to_index[fruit_class])  # Convert string to int

    return np.array(filepaths), np.array(labels)


def preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img


def create_dataset(paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(
        lambda path, label: (preprocess_image(path), label),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


# --- Main Execution ---
if __name__ == "__main__":
    print("Initializing Fruit Classification (Part C)...")

    # 1. Setup Classes
    class_names = get_class_names(train_dir)
    NUM_CLASSES = len(class_names)
    print(f"Detected {NUM_CLASSES} classes: {class_names}")

    # Map class names to integers (0, 1, 2...)
    class_to_index = {name: i for i, name in enumerate(class_names)}

    # 2. Load Data
    print("Loading Training Data...")
    train_paths, train_labels = load_data(train_dir, class_to_index)
    print(f"Training images: {len(train_paths)}")

    print("Loading Validation Data...")
    val_paths, val_labels = load_data(val_dir, class_to_index)
    print(f"Validation images: {len(val_paths)}")

    # 3. Create Pipelines
    train_ds = create_dataset(train_paths, train_labels)
    val_ds = create_dataset(val_paths, val_labels)

    # 4. Build Model with Fine-Tuning
    print("Building MobileNetV2 with Fine-Tuning...")

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=img_size + (3,)
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 5. Training
    print("Starting Training...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    # 6. Save
    model.save("model_part_c_fruit.keras")
    print("Part C Model Saved.")


