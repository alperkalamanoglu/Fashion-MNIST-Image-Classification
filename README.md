# Fashion MNIST Classification with Convolutional Neural Networks

This project uses Convolutional Neural Networks (CNNs) to classify fashion items from the Fashion MNIST dataset. The goal is to accurately identify 10 different classes of clothing items.

## Dataset

The Fashion MNIST dataset consists of 60,000 training images and 10,000 test images, each of size 28x28 pixels. The dataset includes the following classes:

- 0: T-shirt/top
- 1: Trouser
- 2: Pullover
- 3: Dress
- 4: Coat
- 5: Sandal
- 6: Shirt
- 7: Sneaker
- 8: Bag
- 9: Ankle boot

## Model Architecture

The CNN model architecture consists of the following layers:

1. Input layer
2. Convolutional layers with ReLU activation
3. Max pooling layers
4. Flatten layer
5. Fully connected layer with ReLU activation
6. Dropout layer for regularization
7. Output layer with softmax activation

## Training and Evaluation

The model is trained using the Adam optimizer and categorical crossentropy loss. Early stopping is used to prevent overfitting. After training for 50 epochs with a batch size of 64, the model achieves a test accuracy of around 90%.

## Results

The model's performance is evaluated using precision, recall, and F1-score metrics. Additionally, the class distributions, example images from each class, and some incorrect predictions are visualized for better understanding.

## Usage

To train the model, run the provided Python script `fashion_mnist_cnn.py`. Make sure to have the necessary libraries installed, which can be installed using `pip install -r requirements.txt`. You can also use the trained model for inference on new images.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- scikit-learn

## Acknowledgements

This project is based on the Fashion MNIST dataset, which was created by Zalando Research. The dataset is freely available under the https://github.com/zalandoresearch/fashion-mnist

