# Bank Customer Churn Prediction Using ANN

## Overview
This project aims to predict whether a customer will stay or leave a bank using an Artificial Neural Network (ANN). By analyzing customer data, the model provides valuable insights for banks to improve customer retention strategies.

## Dataset
The dataset used for this project is `Churn_Modelling.csv`, which contains information about bank customers, including demographics, account information, and churn status.

[Download Dataset](https://github.com/farhan065/Deep-Learning-Projects/blob/main/Using%20ANN%20to%20estimate%20whether%20a%20customer%20in%20a%20bank%20stays%20or%20leave/Churn_Modelling.csv)

## Features
- **Gender, Age, Tenure, Balance**: Demographic and account details.
- **NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary**: Behavioral features.
- **Geography**: Customer's region.
- **Exited**: Target variable indicating churn (1) or retention (0).

## Steps
1. Data Preprocessing:
   - Handling missing values
   - Encoding categorical variables
   - Feature scaling

2. Model Building:
   - Input layer
   - Hidden layers with ReLU activation
   - Output layer with sigmoid activation

3. Model Training:
   - Optimizer: Adam
   - Loss Function: Binary Crossentropy
   - Metric: Accuracy

4. Model Evaluation:
   - Confusion Matrix
   - Accuracy Score

5. Prediction:
   - Predict churn probability for new customers.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/farhan065/Deep-Learning-Projects.git
   ```
2. Navigate to the project directory:
   ```bash
   cd "Using ANN to estimate whether a customer in a bank stays or leave"
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the script:
   ```bash
   python churn_prediction.py
   ```

---

# Dog vs Cat Classification Using CNN

## Overview
This project uses a Convolutional Neural Network (CNN) to classify images as either a dog or a cat. This deep learning model analyzes image features and achieves high accuracy in recognizing the two classes.

## Dataset
The dataset consists of images of cats and dogs, categorized into training and test sets.

[Download Dataset](https://github.com/farhan065/Deep-Learning-Projects/tree/main/Using%20CNN%20%20to%20identify%20whether%20a%20picture%20represents%20a%20dog%20or%20a%20cat/dataset)

## Features
- **Training Set**: Contains labeled images of cats and dogs.
- **Test Set**: Contains images for model evaluation.

## Steps
1. Data Preprocessing:
   - Image resizing
   - Image augmentation (rotation, flipping, etc.)

2. Model Building:
   - Convolutional layers with ReLU activation
   - MaxPooling layers
   - Fully connected layers
   - Output layer with softmax activation

3. Model Training:
   - Optimizer: Adam
   - Loss Function: Categorical Crossentropy
   - Metrics: Accuracy

4. Model Evaluation:
   - Accuracy and loss curves
   - Confusion Matrix

5. Prediction:
   - Predict class (dog/cat) for new images.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/farhan065/Deep-Learning-Projects.git
   ```
2. Navigate to the project directory:
   ```bash
   cd "Using CNN  to identify whether a picture represents a dog or a cat"
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the script:
   ```bash
   python dog_cat_classifier.py
   ```

## Libraries Used
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib

## Results
- ANN Model: Achieved high accuracy in predicting customer churn.
- CNN Model: Successfully classified dog and cat images with high accuracy.


## Author
[Farhan065](https://github.com/farhan065)
