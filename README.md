# Pneumonia-Prediction
Pneumonia Prediction using Convolutional Neural Networks


This project demonstrates the development and implementation of a Convolutional Neural Network (CNN) to classify chest X-ray images as either Normal or indicative of Pneumonia. The model is built using Keras and optimized with Keras Tuner to achieve high accuracy in medical image analysis.

📋 Project Summary
The goal of this project is to build a robust deep learning model capable of accurately detecting pneumonia from chest X-ray images. This automated approach can serve as a valuable tool for radiologists, potentially speeding up diagnosis and reducing workload. The project covers the entire data science workflow: data loading and preprocessing, data augmentation to enhance model generalization, hyperparameter tuning to find the optimal model architecture, and finally, model evaluation using various performance metrics. The final model achieves an impressive 97.5% accuracy on the test dataset.

📂 Dataset
The dataset consists of chest X-ray images organized into two categories:

PNEUMONIA: Images showing signs of pneumonia.

NORMAL: Images of healthy lungs.

The data is split into training and testing sets, located in separate directories.

🛠️ Methodology
The project follows a structured approach to build the prediction model:

Data Loading: File paths and labels for the train and test images were loaded into Pandas DataFrames for easy management.

Data Augmentation: To prevent overfitting and improve the model's ability to generalize, the training images were augmented using ImageDataGenerator. Transformations included rotation, shifts, shear, zoom, and horizontal flips.

CNN Architecture Design: A flexible CNN architecture was defined with tunable hyperparameters for:

Number of filters in convolutional layers.

Kernel sizes.

Activation functions.

Number of units in dense layers.

Hyperparameter Tuning: Keras Tuner's RandomSearch was employed to automatically find the best combination of hyperparameters, optimizing for accuracy over 5 trials.

Model Training: The best model architecture identified by the tuner was trained for an additional 10 epochs to ensure optimal performance.

Evaluation: The model's performance was evaluated on the unseen test dataset using accuracy, a confusion matrix, and a detailed classification report (precision, recall, F1-score).

Validation: The model was further validated by making predictions on randomly selected images from the test set to provide a qualitative assessment of its performance.

🚀 Technologies Used
Python 3

TensorFlow & Keras: For building and training the CNN.

Keras Tuner: For automated hyperparameter optimization.

Scikit-learn: For model evaluation metrics (Confusion Matrix, Classification Report).

Pandas: For data manipulation and creating dataframes.

NumPy: For numerical operations.

Matplotlib & Seaborn: For data visualization and plotting results.

Google Colab: As the development environment with GPU acceleration.

📈 Results
The optimized model demonstrated excellent performance on the test set:

Overall Accuracy: 97.5%

Loss: 0.185

Classification Report
Class	Precision	Recall	F1-Score	Support
NORMAL	0.95	1.00	0.98	20
PNEUMONIA	1.00	0.95	0.97	20
Total	0.98	0.97	0.97	40

Export to Sheets
Confusion Matrix
The confusion matrix shows that the model made only one error on the entire test set, misclassifying one pneumonia case as normal.
