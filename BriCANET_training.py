import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from pathlib import Path
from src import utils


if __name__ == '__main__':
    # Load dataset
    dataset_directory = r"C:\Users\enzod\Desktop\Dataset_12_12_24_CROP"
    images, class_names = utils.load_dataset_from_directory(dataset_directory)

    # Preprocess data with sliding window and balance patches
    X, y = utils.preprocess_and_balance_patches(images)
    y = to_categorical(y, num_classes=len(class_names))

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Build and Compile Model
    model = utils.build_BriCANET(input_shape=(224, 224, 3), num_classes=len(class_names))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the Model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16)

    # Save the Model
    model.save("BriCANET_trained.h5")

    # Evaluate the Model
    y_pred = model.predict(X_test)
    accuracy, precision, recall, f1 = utils.calculate_metrics(y_test, y_pred)

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
                transformed_img = utils.fuse_color_spaces(img)
                output_path = output_dir / f"example_{class_name}.png"
                cv2.imwrite(str(output_path), cv2.cvtColor(transformed_img, cv2.COLOR_RGB2BGR))
                break

    # Print Metrics
    print(f"Class Names: {class_names}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
