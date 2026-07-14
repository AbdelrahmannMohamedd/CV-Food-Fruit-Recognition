import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, metrics, Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2


dataset_root = "Project Data/Food/Train"
img_size = (128, 128)
batch_size = 32
epochs = 8
max_images_per_class = 10

# ================= DATA =================

def get_food_dict(directory, max_images=10):
    food_dict = {}
    for folder in os.listdir(directory):
        path = os.path.join(directory, folder)
        if os.path.isdir(path):
            images = [
                os.path.join(path, img)
                for img in os.listdir(path)
                if img.lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            images = images[:max_images]

            if len(images) > 1:
                food_dict[folder] = images
    return food_dict


def preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img


def create_triplets(food_dict):
    triplets = []
    classes = list(food_dict.keys())

    for cls in classes:
        imgs = food_dict[cls]

        for i in range(len(imgs) - 1):
            anchor = imgs[i]
            positive = imgs[i + 1]

            neg_cls = cls
            while neg_cls == cls:
                neg_cls = random.choice(classes)

            negative = random.choice(food_dict[neg_cls])
            triplets.append((anchor, positive, negative))

    random.shuffle(triplets)
    return triplets


def get_batch(triplets, batch_size=batch_size):
    for i in range(0, len(triplets), batch_size):
        A, P, N = [], [], []

        for a, p, n in triplets[i:i + batch_size]:
            A.append(preprocess_image(a))
            P.append(preprocess_image(p))
            N.append(preprocess_image(n))

        yield [np.stack(A), np.stack(P), np.stack(N)], None


# ================= MODEL =================

class DistanceLayer(layers.Layer):
    def call(self, a, p, n):
        ap = tf.reduce_sum(tf.square(a - p), axis=-1)
        an = tf.reduce_sum(tf.square(a - n), axis=-1)
        return ap, an


def get_encoder(input_shape):
    base = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )

    for layer in base.layers[:-20]:
        layer.trainable = False

    return Sequential([
        base,
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(128),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ])


class SiameseModel(Model):
    def __init__(self, net, margin=0.5):
        super().__init__()
        self.net = net
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")

    def train_step(self, data):
        x = data[0]

        with tf.GradientTape() as tape:
            ap, an = self.net(x, training=True)
            loss = tf.reduce_mean(tf.maximum(ap - an + self.margin, 0.0))

        grads = tape.gradient(loss, self.net.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.net.trainable_weights))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]


# ================= TRAIN =================

if __name__ == "__main__":
    food_dict = get_food_dict(dataset_root, max_images_per_class)
    triplets = create_triplets(food_dict)

    encoder = get_encoder(img_size + (3,))
    a = layers.Input(img_size + (3,))
    p = layers.Input(img_size + (3,))
    n = layers.Input(img_size + (3,))

    ap, an = DistanceLayer()(encoder(a), encoder(p), encoder(n))
    net = Model([a, p, n], [ap, an])

    model = SiameseModel(net)
    model.compile(optimizer=Adam(1e-4))

    for epoch in range(epochs):
        losses = []
        for batch, _ in get_batch(triplets):
            losses.append(model.train_on_batch(batch))
        print(f"Epoch {epoch+1}: loss = {np.mean(losses):.4f}")

    encoder.save("model_part_b_encoder.keras")
