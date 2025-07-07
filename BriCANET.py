import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense,
                                     Add, Multiply, Activation, BatchNormalization, Flatten, Softmax)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import os
from pathlib import Path
import random

# 1. Color Space Fusion
def fuse_color_spaces(image):
    # Convert to HSV and Lab
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)

    # Equalize the V channel in HSV
    hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])

    # Separate channels
    h, s, v = cv2.split(hsv_image)
    l, a, b = cv2.split(lab_image)

    # Fuse channels
    fused_image = cv2.merge((h + l, s + a, v + b))
    return fused_image

# 2. Attention Mechanism (Squeeze-and-Excitation Module)
def se_block(input_tensor, reduction=16):
    filters = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Dense(filters // reduction, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    return Multiply()([input_tensor, se])

# 3. MBConv Module
def mbconv_block(input_tensor, filters, expansion_factor=6):
    input_filters = input_tensor.shape[-1]

    # Expand
    x = Conv2D(input_filters * expansion_factor, kernel_size=1, padding='same', activation='relu')(input_tensor)
    x = BatchNormalization()(x)

    # Depthwise Convolution
    x = DepthwiseConv2D(kernel_size=3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Squeeze-and-Excitation
    x = se_block(x)

    # Project
    x = Conv2D(filters, kernel_size=1, padding='same')(x)
    x = BatchNormalization()(x)

    # Residual Connection
    if input_filters == filters:
        x = Add()([input_tensor, x])
    return x

# 4. Build MCCA-Net Model
def build_mcca_net(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Initial Convolution
    x = Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)

    # Stacked MBConv Blocks
    x = mbconv_block(x, filters=32)
    x = mbconv_block(x, filters=64)

    # Attention Transformer-like Blocks
    x = mbconv_block(x, filters=128)
    x = mbconv_block(x, filters=128)

    # Classification Head
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, x)

# 5. Sliding Window Preprocessing
def sliding_window(image, window_size, step_size):
    """
    Generate patches from an image using a sliding window.

    Args:
        image (np.array): Input image.
        window_size (tuple): Size of the sliding window (height, width).
        step_size (int): Step size for sliding.

    Returns:
        List of patches (np.array).
    """
    patches = []
    h, w, _ = image.shape
    for y in range(0, h - window_size[1] + 1, step_size):
        for x in range(0, w - window_size[0] + 1, step_size):
            patch = image[y:y + window_size[1], x:x + window_size[0]]
            patches.append(patch)
    return patches

# 6. Data Augmentation
def augment_patch(patch):
    """
    Apply random transformations to a patch to create augmented versions.

    Args:
        patch (np.array): Input image patch.

    Returns:
        Augmented patch.
    """
    # Random horizontal flip
    if random.random() > 0.5:
        patch = cv2.flip(patch, 1)

    # Random brightness adjustment
    if random.random() > 0.5:
        factor = 1.0 + (random.random() - 0.5) * 0.4  # Adjust brightness by ±20%
        patch = np.clip(patch * factor, 0, 1)

    # Random rotation (90 degrees steps)
    if random.random() > 0.5:
        k = random.randint(0, 3)
        patch = np.rot90(patch, k)

    return patch

# 7. Preprocess Images and Balance Patches
def preprocess_and_balance_patches(images, window_size=(224, 224), step_size=112):
    patches = []
    labels = []

    class_patches = {}

    # Create patches per class
    for img, label in images:
        fused_img = fuse_color_spaces(img)
        img_patches = sliding_window(fused_img, window_size, step_size)
        if label not in class_patches:
            class_patches[label] = []
        class_patches[label].extend(img_patches)

    # Find the maximum number of patches per class
    max_patches = max(len(patch_list) for patch_list in class_patches.values())

    # Balance patches by augmentation
    for label, patch_list in class_patches.items():
        # Add original patches
        patches.extend(patch_list)
        labels.extend([label] * len(patch_list))

        # Augment and add patches to balance the dataset
        while len(patch_list) < max_patches:
            patch = random.choice(patch_list)
            augmented_patch = augment_patch(patch)
            patches.append(augmented_patch)
            labels.append(label)
            patch_list.append(augmented_patch)  # Update the list to include the new patch

    return np.array(patches) / 255.0, np.array(labels)

# 8. Load Dataset
def load_dataset_from_directory(directory):
    class_names = os.listdir(directory)
    class_indices = {name: idx for idx, name in enumerate(class_names)}

    images = []
    labels = []

    for class_name, class_idx in class_indices.items():
        class_dir = Path(directory) / class_name
        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:  # Solo immagini
                img = cv2.imread(str(img_path))
                if img is not None:  # Assicurati che l'immagine sia valida
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append((img, class_idx))

    return images, class_names

# 9. Metrics Calculation
def calculate_metrics(y_true, y_pred):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Load dataset
dataset_directory = r"C:\Users\enzod\Desktop\Dataset_12_12_24_CROP"
images, class_names = load_dataset_from_directory(dataset_directory)

# Preprocess data with sliding window and balance patches
X, y = preprocess_and_balance_patches(images)
y = to_categorical(y, num_classes=len(class_names))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build and Compile Model
model = build_mcca_net(input_shape=(224, 224, 3), num_classes=len(class_names))
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16)

# Save the Model
model.save("mcca_net_model.h5")

# Evaluate the Model
y_pred = model.predict(X_test)
accuracy, precision, recall, f1 = calculate_metrics(y_test, y_pred)

# Plot Confusion Matrix
y_true = np.argmax(y_test, axis=1)
y_pred_classes = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig("confusion_matrix.png")
plt.close()

# Plot Training History
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training History")
plt.xlabel("Epochs")
plt.ylabel("Accuracy / Loss")
plt.legend()
plt.savefig("training_history.png")
plt.close()

# Save Example Transformed Images
output_dir = Path("transformed_examples")
output_dir.mkdir(exist_ok=True)
for class_name, class_idx in zip(class_names, range(len(class_names))):
    for img, label in images:
        if label == class_idx:
            transformed_img = fuse_color_spaces(img)
            output_path = output_dir / f"example_{class_name}.png"
            cv2.imwrite(str(output_path), cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR))
            break

# Print Metrics
print(f"Class Names: {class_names}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
