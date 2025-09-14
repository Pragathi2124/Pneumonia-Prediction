# Pneumonia-Prediction
## Pneumonia Prediction using Convolutional Neural Networks

##### Pneumonia is a significant global health challenge, responsible for high rates of morbidity and mortality, particularly among vulnerable populations. Traditional diagnosis via clinical evaluation and radiological interpretations of chest X-rays can be time-consuming, subjective, and limited by human variability. There is an urgent need for automated, accurate, and scalable diagnostic solutions to improve early detection and treatment outcomes.

##### This project aims to develop a deep learning-based system for automated classification of chest X-ray images into 'Normal' or 'Pneumonia' categories, utilizing Convolutional Neural Networks (CNNs) and hyperparameter optimization techniques. By leveraging medical image analysis, the objective is to enhance diagnostic accuracy, reduce dependency on manual interpretation, and support clinical decision-making for pneumonia detection


**📋 Project Summary**
The goal of this project is to build a robust deep learning model capable of accurately detecting pneumonia from chest X-ray images. This automated approach can serve as a valuable tool for radiologists, potentially speeding up diagnosis and reducing workload. The project covers the entire data science workflow: data loading and preprocessing, data augmentation to enhance model generalization, hyperparameter tuning to find the optimal model architecture, and finally, model evaluation using various performance metrics. The final model achieves an impressive 97.5% accuracy on the test dataset.

**📂 Dataset**
The dataset consists of chest X-ray images organized into two categories:
PNEUMONIA: Images showing signs of pneumonia.
NORMAL: Images of healthy lungs.
The data is split into training and testing sets, located in separate directories.

**🛠️ Methodology**
The project follows a structured approach to build the prediction model:
**1.Data Loading**: File paths and labels for the train and test images were loaded into Pandas DataFrames for easy management.

**2.Data Augmentation**: To prevent overfitting and improve the model's ability to generalize, the training images were augmented using ImageDataGenerator. Transformations included rotation, shifts, shear, zoom, and horizontal flips.

**3.CNN Architecture Design**: A flexible CNN architecture was defined with tunable hyperparameters for:
Number of filters in convolutional layers.
Kernel sizes.
Activation functions.
Number of units in dense layers.

**4.Hyperparameter Tuning**: Keras Tuner's RandomSearch was employed to automatically find the best combination of hyperparameters, optimizing for accuracy over 5 trials.

**5.Model Training**: The best model architecture identified by the tuner was trained for an additional 10 epochs to ensure optimal performance.

**6.Evaluation**: The model's performance was evaluated on the unseen test dataset using accuracy, a confusion matrix, and a detailed classification report (precision, recall, F1-score).

**7.Validation:** The model was further validated by making predictions on randomly selected images from the test set to provide a qualitative assessment of its performance.

### 🚀 Technologies Used
Python 3
TensorFlow & Keras: For building and training the CNN.
Keras Tuner: For automated hyperparameter optimization.
Scikit-learn: For model evaluation metrics (Confusion Matrix, Classification Report).
Pandas: For data manipulation and creating dataframes.
NumPy: For numerical operations.
Matplotlib & Seaborn: For data visualization and plotting results.
Google Colab: As the development environment with GPU acceleration.

### 📈 Results
The optimized model demonstrated excellent performance on the test set:
Overall Accuracy: 97.5%
Loss: 0.185
Classification Report
Class   	Precision	 Recall	 F1-Score	 Support
NORMAL	    0.95	    1.00	   0.98	     20
PNEUMONIA	  1.00	    0.95	   0.97    	 20
Total      	0.98	    0.97	   0.97	     40

**Confusion Matrix**
The confusion matrix shows that the model made only one error on the entire test set, misclassifying one pneumonia case as normal.
