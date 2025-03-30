import tensorflow as tf
from transformers import TFCLIPModel, CLIPProcessor
from collections import defaultdict
import tensorflow_datasets as tfds
import numpy as np
from PIL import Image
import random
import os

# Set seeds for reproducibility
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load dataset and class names
dataset = tfds.load("tf_flowers", split="train", as_supervised=True)
label_names = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']

# Load CLIP model and processor
model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Prepare test samples
all_test_samples = list(tfds.as_numpy(dataset))
test_samples = all_test_samples[1000:1400]
test_images = [img for img, lbl in test_samples]
test_labels = [label_names[lbl] for img, lbl in test_samples]
prompts = [f"This is a photo of a {label}" for label in test_labels]

# Prepare text features (frozen text encoder)
text_inputs = processor(text=prompts, return_tensors="tf", padding=True)
text_features = model.get_text_features(**text_inputs)
text_features = tf.nn.l2_normalize(text_features, axis=-1)

correct = 0
for img, true_label in zip(test_images, test_labels):
    inputs = processor(images=img, return_tensors="tf", padding=True)
    image_features = model.get_image_features(**inputs)
    image_features = tf.nn.l2_normalize(image_features, axis=-1)
    sims = tf.matmul(image_features, text_features, transpose_b=True)
    pred_idx = tf.argmax(sims, axis=1).numpy()[0]
    pred_label = test_labels[pred_idx]
    if pred_label == true_label:
        correct += 1
print(f"\nAccuracy on test set1: {correct}/{len(test_images)} = {correct/len(test_images):.2%}")

# Freeze all model layers
for var in model.variables:
    var._trainable = False

# Unfreeze the first LayerNorm in the vision transformer encoder
first_block = model.clip.vision_model.encoder.layers[0]
for layer in first_block.submodules:
    if isinstance(layer, tf.keras.layers.LayerNormalization):
        for var in layer.variables:
            var._trainable = True
        break

trainable_vars = [v for v in model.variables if v.trainable]

# Sample imbalanced training data
num_classes = len(label_names)
proportions = np.random.dirichlet([0.1] * num_classes)
sample_counts = [1 + int(np.round(p * (16 - 1))) for p in proportions]

class_counter = defaultdict(int)
train_images, train_labels = [], []

for image, label in tfds.as_numpy(dataset):
    label_str = label_names[label]
    if class_counter[label_str] < sample_counts[label]:
        train_images.append(image)
        train_labels.append(label)
        class_counter[label_str] += 1
    if all(class_counter[label_names[i]] >= sample_counts[i] for i in range(num_classes)):
        break

# Prepare text prompts for each class
class_prompts = [f"a photo of a {label}" for label in label_names]
text_inputs = processor(text=class_prompts, return_tensors="tf", padding=True)
class_text_features = model.get_text_features(**text_inputs)
class_text_features = tf.nn.l2_normalize(class_text_features, axis=-1)

# Compute LDAM margins
def get_ldam_margins(class_counts, max_m=0.5, s=30):
    margins = 1.0 / np.sqrt(np.sqrt(class_counts))
    margins = margins * (max_m / np.max(margins))
    return tf.convert_to_tensor(margins, dtype=tf.float32), s

class_counts_dict = defaultdict(int)
for lbl in train_labels:
    class_counts_dict[lbl] += 1
class_counts = [class_counts_dict[i] for i in range(num_classes)]
ldam_margins, scale_s = get_ldam_margins(class_counts)

# Define LDAM loss
def ldam_loss(logits, labels, margins, scale=30):
    labels_onehot = tf.one_hot(labels, depth=tf.shape(logits)[-1])
    margins_applied = labels_onehot * margins
    logits_m = logits - margins_applied
    logits_m *= scale
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits_m, from_logits=True)

# Prepare training data
train_prompts = [f"a photo of a {label_names[label]}" for label in train_labels]
train_inputs = processor(text=train_prompts, images=train_images, return_tensors="tf", padding=True)

# Build dataset
batch_size = 16
dataset = tf.data.Dataset.from_tensor_slices((train_inputs['pixel_values'], train_labels)).batch(batch_size)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Training step
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        outputs = model.get_image_features(pixel_values=images)
        image_embeds = tf.nn.l2_normalize(outputs, axis=-1)
        logits = tf.matmul(image_embeds, class_text_features, transpose_b=True)
        loss = ldam_loss(logits, labels, ldam_margins, scale=scale_s)
        loss = tf.reduce_mean(loss)
    grads = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(grads, trainable_vars))
    return loss

# Train
for epoch in range(30):
    epoch_loss = 0
    for batch_images, batch_labels in dataset:
        loss = train_step(batch_images, batch_labels)
        epoch_loss += loss
    epoch_loss /= len(dataset)
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss.numpy():.4f}")

# Inference
text_inputs = processor(text=class_prompts, return_tensors="tf", padding=True)
class_text_features = model.get_text_features(**text_inputs)
class_text_features = tf.nn.l2_normalize(class_text_features, axis=-1)

correct = 0
for img, true_label in zip(test_images, test_labels):
    inputs = processor(images=img, return_tensors="tf", padding=True)
    image_features = model.get_image_features(**inputs)
    image_features = tf.nn.l2_normalize(image_features, axis=-1)
    sims = tf.matmul(image_features, class_text_features, transpose_b=True)
    pred_idx = tf.argmax(sims, axis=1).numpy()[0]
    pred_label = label_names[pred_idx]
    if pred_label == true_label:
        correct += 1

print(f"\nAccuracy on test set: {correct}/{len(test_images)} = {correct / len(test_images):.2%}")
