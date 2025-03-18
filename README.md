# Fashion-MNIST-Classification

This repository contains a deep learning model built using TensorFlow and Keras to classify images from the Fashion MNIST dataset. The model is implemented in a Jupyter Notebook and employs a Convolutional Neural Network (CNN) for image classification.


Dataset
The Fashion MNIST dataset consists of 70,000 grayscale images of 10 different fashion categories, with 60,000 images for training and 10,000 images for testing.
Features
•	Uses a CNN with six layers to classify images.
•	Implements TensorFlow and Keras for deep learning.
•	Visualizes dataset samples and model predictions.


Installation
Ensure you have Python and Jupyter Notebook installed. You can install the required dependencies using:
pip install tensorflow numpy matplotlib
Usage
1.	Clone the repository:
git clone https://github.com/Alims-jay/fashion-mnist-classification.git
2.	Navigate to the project directory:
cd fashion-mnist-classification
3.	Launch Jupyter Notebook:
jupyter notebook
4.	Open and run the Fashion MNIST Classification.ipynb notebook.

   
Model Architecture
The CNN model consists of:
•	Convolutional layers with ReLU activation
•	MaxPooling layers for feature reduction
•	Fully connected dense layers
•	Softmax activation for multi-class classification


Results
The trained model achieves high accuracy in classifying fashion items. The notebook includes visualizations of model predictions.
