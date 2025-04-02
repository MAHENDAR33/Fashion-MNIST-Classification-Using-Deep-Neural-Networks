# Fashion-MNIST-Classification-Using-Deep-Neural-Networks
Introduction
This project aims to classify images from the Fashion MNIST dataset using a deep neural network. The dataset consists of 70,000 grayscale images (28x28 pixels) categorized into 10 classes representing different clothing items such as T-shirts, trousers, sneakers, and coats. The model is trained to recognize these categories using a multi-layer neural network.
Dataset and Preprocessing
•	The dataset is loaded and split into training (60,000 images) and testing (10,000 images) sets.
•	Images are normalized by scaling pixel values to the range [0,1] to improve model performance.
•	Labels are converted to one-hot encoded format for categorical classification.
Model Architecture
The neural network consists of the following layers:
1.	Flatten Layer: Converts 28x28 images into a 1D array.
2.	Dense Layer (512 neurons, ReLU activation): First hidden layer for feature extraction.
3.	Dense Layer (256 neurons, ReLU activation): Second hidden layer to refine extracted features.
4.	Dense Layer (128 neurons, ReLU activation): Third hidden layer for deeper learning.
5.	Dense Layer (10 neurons, Softmax activation): Output layer for classification into 10 categories.
Training and Optimization
•	The model is compiled using the Adam optimizer and categorical crossentropy loss function.
•	The dataset is trained for 15 epochs with a batch size of 32.
•	TensorBoard logging is implemented to monitor training progress.
Evaluation and Results
•	The model achieved high accuracy on the test dataset, indicating good generalization.
•	The final model is saved for future predictions and reusability.
•	A plot of training and validation accuracy over epochs is generated to visualize performance trends.
Model Validation and Consistency Check
•	The saved model is reloaded, and its structure and parameters are compared with the original model.
•	Model weights are verified for consistency, ensuring no data corruption during saving and loading.
Prediction on New Images
•	A function is implemented to predict and visualize classifications on test images.
•	The model accurately predicts the class of a given image, validating its effectiveness.
Conclusion
This project successfully classifies Fashion MNIST images using a deep neural network. The model demonstrates high accuracy and consistency in recognizing different clothing categories. The implementation includes data preprocessing, model training, validation, and prediction functions, making it a comprehensive approach to image classification using deep learning.

