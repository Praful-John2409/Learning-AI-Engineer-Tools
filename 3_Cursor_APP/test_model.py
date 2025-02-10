import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

def load_and_preprocess_image(image_path):
    # Load image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    # Resize to 28x28
    img = img.resize((28, 28))
    # Convert to numpy array
    img_array = np.array(img)
    # Invert if necessary (MNIST has white digits on black background)
    if img_array.mean() > 128:
        img_array = 255 - img_array
    return img_array

def test_custom_image(model_path, image_path):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Load and preprocess the image
    img_array = load_and_preprocess_image(image_path)
    
    # Get prediction
    img_for_prediction = img_array.reshape(1, 784)
    img_for_prediction = img_for_prediction.astype('float32') / 255.0
    
    prediction = model.predict(img_for_prediction, verbose=0)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    # Display results
    plt.figure(figsize=(4, 4))
    plt.imshow(img_array, cmap='gray')
    plt.title(f'Predicted: {predicted_digit}\nConfidence: {confidence:.1f}%')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # If files are in the same folder, just use the filenames
    test_custom_image('mnist_model.h5', 'digit.png')  # Replace 'digit.png' with your image filename 