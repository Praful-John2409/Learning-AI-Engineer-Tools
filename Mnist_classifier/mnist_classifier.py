import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Data Loading and Preprocessing
def load_and_preprocess_data():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0,1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape images to 1D arrays
    x_train = x_train.reshape((-1, 28*28))
    x_test = x_test.reshape((-1, 28*28))
    
    # One-hot encode labels
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

# 2. Create Model
def create_model():
    model = models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    return model

# 3. Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    # Convert one-hot encoded labels back to class numbers
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Plot using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def predict_single_image(model, image):
    # Ensure image is in correct format (1D array of 784 pixels)
    image = image.reshape(1, 784)
    # Normalize pixel values
    image = image.astype('float32') / 255.0
    
    # Get prediction
    prediction = model.predict(image, verbose=0)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    return predicted_digit, confidence

def test_random_samples(model, x_test, y_test, num_samples=5):
    # Select random samples
    indices = np.random.randint(0, len(x_test), num_samples)
    
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        # Get the image and true label
        image = x_test[idx].reshape(28, 28)
        true_digit = np.argmax(y_test[idx])
        
        # Get prediction
        predicted_digit, confidence = predict_single_image(model, x_test[idx])
        
        # Plot
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'Pred: {predicted_digit}\nTrue: {true_digit}\nConf: {confidence:.1f}%')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('test_samples.png')
    plt.close()

def main():
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    
    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # 2. Create and compile model
    print("Creating model...")
    model = create_model()
    
    # 3. Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 4. Model summary
    model.summary()
    
    # 5. Train model
    print("\nTraining model...")
    batch_size = 128
    epochs = 20
    
    # Split training data into train and validation sets
    validation_split = 0.1
    
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        verbose=1
    )
    
    # 6. Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # 7. Generate predictions and metrics
    y_pred = model.predict(x_test)
    
    # Print classification report
    print("\nClassification Report:")
    y_true_classes = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print(classification_report(y_true_classes, y_pred_classes))
    
    # 8. Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    print("\nConfusion matrix has been saved as 'confusion_matrix.png'")
    
    # 9. Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    print("Training history has been saved as 'training_history.png'")

    # After model evaluation, add:
    print("\nTesting random samples...")
    test_random_samples(model, x_test, y_test)
    print("Test samples have been saved as 'test_samples.png'")
    
    # Save the model for later use
    model.save('mnist_model.h5')
    print("Model has been saved as 'mnist_model.h5'")

if __name__ == "__main__":
    main() 