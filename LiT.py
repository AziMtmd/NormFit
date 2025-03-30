import torch
from torch.utils.data import DataLoader
import tensorflow_datasets as tfds
import numpy as np
import random
import open_clip
from PIL import Image
from collections import defaultdict
import sys

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load OpenCLIP model (LiT-style)
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name='ViT-L-14',
    pretrained='laion2b_s32b_b82k',
    device=device
)
tokenizer = open_clip.get_tokenizer('ViT-L-14')

# Load dataset
ds = tfds.load("tf_flowers", split='train', as_supervised=True)
data = list(tfds.as_numpy(ds))

label_names = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']

all_test_samples = list(tfds.as_numpy(ds))
print('all_test_samples', len(all_test_samples))
test_samples = all_test_samples[1000:3500]
test_images = [img for img, lbl in test_samples]
test_labels = [lbl for img, lbl in test_samples]  # integer labels
prompts2 = [f"This is a photo of a {label_names[lbl]}" for lbl in test_labels]

num_classes = len(label_names)
proportions = np.random.dirichlet([0.1] * num_classes)
sample_counts = [1 + int(np.round(p * (100 - 1))) for p in proportions]
print("Desired sample counts per class:")
for i, count in enumerate(sample_counts):
    print(f"  {label_names[i]}: {count}")

class_counter = defaultdict(int)
train_images, train_labels = [], []

for image, label in tfds.as_numpy(ds):
    label_str = label_names[label]
    if class_counter[label_str] < sample_counts[label]:
        train_images.append(image)
        train_labels.append(label_str)
        class_counter[label_str] += 1
    if all(class_counter[label_names[i]] >= sample_counts[i] for i in range(num_classes)):
        break
print("Actual samples per class:")
for label in label_names:
    print(f"  {label}: {class_counter[label]}")

train_prompts = [f"a photo of a {label}" for label in train_labels]

class LiTDataset(torch.utils.data.Dataset):
    def __init__(self, train_images, train_prompts):
        self.data = list(zip(train_images, train_prompts))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, prompt = self.data[idx]
        return Image.fromarray(image), prompt

class LiTDatasetTest(torch.utils.data.Dataset):
    def __init__(self, test_images, test_labels):
        self.data = list(zip(test_images, test_labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        return Image.fromarray(image), label

def collate_fn(batch):
    images, prompts = zip(*batch)
    images = torch.stack([preprocess(img) for img in images])
    texts = tokenizer(list(prompts))
    return {"image": images, "text": texts}

def collate_fn2(batch):
    images, labels = zip(*batch)
    images = torch.stack([preprocess(img) for img in images])
    labels = torch.tensor(labels)
    return {"image": images, "labels": labels}

train_loader = DataLoader(LiTDataset(train_images, train_prompts), batch_size=16, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(LiTDatasetTest(test_images, test_labels), batch_size=16, collate_fn=collate_fn2)

# Prepare class text features
class_prompts = [f"a photo of a {label}" for label in label_names]
text_inputs = tokenizer(class_prompts)
text_inputs = text_inputs.to(device)
text_features = model.encode_text(text_inputs)
text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# Evaluation
model.eval()
correct = total = 0
with torch.no_grad():
    for batch in test_loader:
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)
        image_embeds = model.encode_image(images)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        logits = image_embeds @ text_features.T
        preds = logits.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)
print(f"Accuracy1: {correct}/{total} = {correct/total:.2%}")

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze ONLY first LayerNorm in image encoder
unfrozen = False
for name, module in model.visual.named_modules():
    if isinstance(module, torch.nn.LayerNorm) and not unfrozen:
        for param in module.parameters():
            param.requires_grad = True
            print(f"✅ Unfroze LayerNorm: {name}")
        unfrozen = True

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-3)

# Training loop
model.train()
for epoch in range(20):
    epoch_loss = 0
    for batch in train_loader:
        images = batch["image"].to(device)
        texts = batch["text"].to(device)
        image_embeds = model.encode_image(images)
        text_embeds = model.encode_text(texts)

        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        logits = image_embeds @ text_embeds.t()
        labels = torch.arange(len(logits)).to(device)

        loss = torch.nn.functional.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")

prompts2 = [f"a photo of a {label_names[lbl]}" for lbl in test_labels]

# Evaluation
model.eval()
correct = total = 0
with torch.no_grad():
    for batch in test_loader:
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)
        image_embeds = model.encode_image(images)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        logits = image_embeds @ text_features.T
        preds = logits.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Accuracy2: {correct}/{total} = {correct/total:.2%}")