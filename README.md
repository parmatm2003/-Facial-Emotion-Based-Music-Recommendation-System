# Facial-Emotion-Based-Music-Recommendation-System
SAP-FER (SAP Facial Emotion Recognition) is a custom deep learning architecture designed for real-time facial emotion recognition and personalized music recommendation. The model leverages a Convolutional Neural Network (CNN) structure combined with advanced techniques such as batch normalization and dropout layers, specifically tuned for emotion detection tasks.

Key Features:

1. Custom CNN Model: SAP-FER is built using a custom CNN architecture that includes multiple convolutional layers, pooling layers, and dense layers to effectively capture and analyze facial features from images.

2. Batch Normalization: Unlike traditional models like VGG16, SAP-FER incorporates batch normalization after each convolutional and dense layer. This technique helps in stabilizing and speeding up the training process by normalizing the input to each layer.

3. Dropout Regularization: The architecture includes strategically placed dropout layers to prevent overfitting, ensuring the model generalizes well to unseen data.

4. Emotion Detection: SAP-FER is specifically designed for facial emotion recognition, classifying emotions into seven categories: Happy, Neutral, Sad, Angry, Disgusted, Surprised, and Fearful.

5. Real-Time Processing: The model is optimized for real-time video input, making it suitable for applications where instant emotion detection is required.

6. Music Recommendation Integration: SAP-FER is integrated with the Spotify API, allowing it to recommend personalized music based on the detected emotions in real-time.

7. High Accuracy: The model achieved an accuracy of 88.43% on the training set, demonstrating its effectiveness in recognizing subtle facial expressions.

Architecture Details:

1. Input Layer: Processes grayscale images with a size of 48x48 pixels, fed into the network in a 3D structure representing width, height, and color channels.

2. Convolutional Layers:

32 filters, 64 filters, and 128 filters, each with a kernel size of (3, 3) and ReLU activation.
Each convolutional layer is followed by batch normalization to maintain the stability of the learning process.
Pooling Layers: MaxPooling layers with a pool size of (2, 2) are applied after specific convolutional layers to reduce spatial dimensions while preserving essential features.

3. Dropout Layers: Dropout layers with a rate of 0.25 are included after certain layers to mitigate overfitting by randomly dropping units during training.

4. Dense Layers:
A high-dimensional dense layer with 2048 units and ReLU activation is used.
L2 kernel regularization is applied to prevent overfitting.
The final dense layer consists of 7 neurons with a softmax activation function, outputting the probabilities of each emotion class.
Optimization:

5. Loss Function: Categorical cross-entropy is used for multi-class classification.
Optimizer: Adam optimizer with a learning rate of 0.001 is used for efficient gradient-based learning.
Training Process: The model was trained for 100 epochs with a batch size of 64, using the FER2013 dataset, which includes 28,709 training images and 7,178 validation images.

6. Custom Callbacks: A custom log callback (LogCallback) is employed to monitor training progress and log loss and accuracy metrics after each epoch.

Dataset:

1. FER2013 Dataset: The model was trained and validated using the FER2013 dataset, which contains 35,887 grayscale images categorized into seven emotions.

2. Spotify Integration: The model utilizes the Spotify Web API to fetch playlists corresponding to the detected emotions, ensuring that music recommendations are tailored to the user's mood.