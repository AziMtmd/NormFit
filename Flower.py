import tensorflow as tf
from transformers import TFCLIPModel, CLIPProcessor
from collections import defaultdict
import tensorflow_datasets as tfds
import sys
import numpy as np
from PIL import Image
import random 
import os

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

dataset = tfds.load("tf_flowers", split=["train"], as_supervised=True)[0]
label_names = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']
model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

all_test_samples = list(tfds.as_numpy(dataset))
test_samples = all_test_samples[1000:1400]
test_images = [img for img, lbl in test_samples]
test_labels = [label_names[lbl] for img, lbl in test_samples]
prompts = [f"This is a photo of a {label}" for label in test_labels]

# Prepare text features (frozen text encoder)
text_inputs = processor(text=prompts, return_tensors="tf", padding=True)
text_features = model.get_text_features(**text_inputs)
text_features = tf.nn.l2_normalize(text_features, axis=-1)

# Inference PAK
# correct = 0
# for img, true_label in zip(test_images, test_labels):
#     inputs = processor(images=img, return_tensors="tf", padding=True)
#     image_features = model.get_image_features(**inputs)
#     image_features = tf.nn.l2_normalize(image_features, axis=-1)
#     sims = tf.matmul(image_features, text_features, transpose_b=True)
#     pred_idx = tf.argmax(sims, axis=1).numpy()[0]
#     pred_label = test_labels[pred_idx]
#     if pred_label == true_label:
#         correct += 1
# print(f"\nAccuracy on test set1: {correct}/{len(test_images)} = {correct/len(test_images):.2%}")
#PAK
# Run a forward pass to build weights
dummy_image = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
inputs = processor(images=dummy_image, text=["a flower"], return_tensors="tf", padding=True)
_ = model(**inputs)

# Step 2: Fully disable training on all variables
def freeze_all_layers(model):
    for var in model.variables:
        var._trainable = False

freeze_all_layers(model)

# Step 4: Unfreeze ONLY the first LayerNorm in the first vision transformer block
# print('len', len(model.clip.text_model.encoder.layers))
# first_block = model.clip.text_model.encoder.layers[0:11]
# unfrozen = 0
# for i in range(len(first_block)):
#     for layer in first_block[i].submodules:
#         if isinstance(layer, tf.keras.layers.LayerNormalization):
#             for var in layer.variables:
#                 var._trainable = True
#                 print(f"✅ Unfroze: {var.name}")
#             unfrozen += 1
#             break  # Stop after first LayerNorm
#     if unfrozen == 0:
#         print("❌ No LayerNorm was unfrozen!")

print('len', len(model.clip.vision_model.encoder.layers))
first_block = model.clip.vision_model.encoder.layers[0]
unfrozen = 0
# for i in range(len(first_block)):
for layer in first_block.submodules:
    if isinstance(layer, tf.keras.layers.LayerNormalization):
        for var in layer.variables:
            var._trainable = True
            print(f"✅ Unfroze: {var.name}")
        unfrozen += 1
        break  # Stop after first LayerNorm
if unfrozen == 0:
    print("❌ No LayerNorm was unfrozen!")

# Step 5: Confirm only those variables are trainable
trainable_vars = [v for v in model.variables if v.trainable]
print(f"\n✅ Total trainable variables: {len(trainable_vars)}")
for v in trainable_vars:
    print(v.name, v.shape)
# sys.exit()
num_classes = len(label_names)
# Draw proportions from the Dirichlet distribution.
proportions = np.random.dirichlet([0.1] * num_classes)
# Convert proportions to counts in the range [1, 16]
# (using 1 + round(p * 15) so that if p ~ 1, we get 16; if p ~ 0, we get at least 1).
sample_counts = [1 + int(np.round(p * (2 - 1))) for p in proportions]
print("Desired sample counts per class:")
for i, count in enumerate(sample_counts):
    print(f"  {label_names[i]}: {count}")

# Collect training samples based on these counts.
class_counter = defaultdict(int)
train_images, train_labels = [], []

for image, label in tfds.as_numpy(dataset):
    label_str = label_names[label]
    # Use the integer label as index to sample_counts.
    if class_counter[label_str] < sample_counts[label]:
        train_images.append(image)
        train_labels.append(label_str)
        class_counter[label_str] += 1
    # Once we have enough samples for every class, stop.
    if all(class_counter[label_names[i]] >= sample_counts[i] for i in range(num_classes)):
        break
print("Actual samples per class:")
for label in label_names:
    print(f"  {label}: {class_counter[label]}")

train_prompts = [f"a photo of a {label}" for label in train_labels]

train_inputs = processor(
    text=train_prompts,
    images=train_images,
    return_tensors="tf",
    padding=True
)

# Compile the model (we only train the LayerNorm, so use low LR)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# loss_fn = tf.keras.losses.CosineSimilarity(axis=-1)

def focal_loss(logits, labels, gamma=1.0, alpha=0.25):
    # Convert logits to softmax probabilities
    probs = tf.nn.softmax(logits, axis=-1)
    # Gather the probability corresponding to the correct class
    labels_onehot = tf.one_hot(labels, depth=tf.shape(logits)[-1])
    pt = tf.reduce_sum(labels_onehot * probs, axis=-1)
    # Compute focal loss
    loss = -alpha * tf.pow(1. - pt, gamma) * tf.math.log(tf.clip_by_value(pt, 1e-9, 1.))
    return loss

def clip_contrastive_loss(image_embeds, text_embeds, logit_scale, gamma=1.0, alpha=0.25):
    # Normalize embeddings
    
    image_embeds = tf.nn.l2_normalize(image_embeds, axis=-1)
    text_embeds = tf.nn.l2_normalize(text_embeds, axis=-1)

    # Compute cosine similarity matrix
    logits_per_image = tf.matmul(image_embeds, text_embeds, transpose_b=True)
    logits_per_text = tf.transpose(logits_per_image)

    # Apply temperature scaling
    logits_per_image *= logit_scale
    logits_per_text *= logit_scale

    # Ground truth: diagonal match
    batch_sizei = tf.shape(logits_per_image)[0]
    labelsi = tf.range(batch_sizei)

    # Use focal loss instead of cross-entropy
    loss_i2t = focal_loss(logits_per_image, labelsi, gamma=gamma, alpha=alpha)
    loss_t2i = focal_loss(logits_per_text, labelsi, gamma=gamma, alpha=alpha)

    # return (tf.reduce_mean(loss_i2t) + tf.reduce_mean(loss_t2i)) / 2
    return tf.reduce_mean(loss_i2t + loss_t2i) / 2

batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(train_inputs).batch(batch_size)

# Forward pass of CLIP returns image_embeds and text_embeds, use dot-product or cosine for similarity
@tf.function
def train_step(inputs):
    with tf.GradientTape() as tape:
        outputs = model(**inputs)
        image_embeds = tf.nn.l2_normalize(outputs.image_embeds, axis=-1)
        text_embeds = tf.nn.l2_normalize(outputs.text_embeds, axis=-1)
        logit_scale = tf.exp(model.clip.logit_scale)
        loss = clip_contrastive_loss(image_embeds, text_embeds, logit_scale)
    grads = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(grads, trainable_vars))
    return loss

# Train multiple epochs
for epoch in range(30):
    epoch_loss = 0
    for batch in dataset:
        batch_loss = train_step(batch)
        epoch_loss += batch_loss
    epoch_loss /= len(dataset)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss.numpy():.4f}")

prompts = [f"a photo of a {label}" for label in test_labels]

# Prepare text features (frozen text encoder)
text_inputs = processor(text=prompts, return_tensors="tf", padding=True)
text_features = model.get_text_features(**text_inputs)
text_features = tf.nn.l2_normalize(text_features, axis=-1)
# Inference
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

print(f"\nAccuracy on test set: {correct}/{len(test_images)} = {correct/len(test_images):.2%}")
