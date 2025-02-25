# Deep Learning Course

![Python Logo](https://www.python.org/static/community_logos/python-logo.png)

## Overview
This repository contains practical implementations of Machine Learning and Deep Learning concepts using Python libraries. The content is structured into multiple sections, each covering a key aspect of data analysis and model building.

---

## Contents

### 1. Introduction to Deep Learning
- **File:** Deep Learning intro.ppsx
- **Description:** A PowerPoint presentation providing an overview of Deep Learning concepts, including neural networks, activation functions, and training methodologies.

### 2. Multi-Layer Perceptron (MLP) for Cell Images
- **File:** mlp-cell-images.ipynb
- **Description:** A Jupyter Notebook implementing an MLP model for classifying cell images. This includes data preprocessing, model architecture definition, training, and evaluation.
- **Dataset:** [Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria)
- **Key Topics:**
  - Data loading and preprocessing
  - Model architecture (fully connected layers, activation functions)
  - Training using backpropagation and optimization algorithms
  - Performance evaluation and visualization

### 3. Multi-Layer Perceptron (MLP) for MNIST
- **File:** mlp-mnist.ipynb
- **Description:** This notebook demonstrates training an MLP model on the MNIST dataset for digit classification. It covers data handling, neural network implementation, and performance metrics.
- **Key Topics:**
  - Dataset loading (MNIST handwritten digits)
  - Neural network structure and hyperparameters
  - Training process and accuracy evaluation
  - Results visualization (loss curves, accuracy plots)

### 4. Vegetables Classification Model
- **File:** vegetables-classification-model.ipynb
- **Description:** A deep learning model for classifying different types of vegetables using convolutional neural networks (CNNs). The notebook walks through dataset handling, model architecture, training, and performance evaluation.
- **Dataset:** [Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset)
- **Key Topics:**
  - Image preprocessing and augmentation
  - CNN architecture and layers
  - Model training and validation
  - Accuracy and confusion matrix analysis

### 5. COVID-19 Detection using Deep Learning
- **File:** covid.ipynb
- **Description:** This notebook applies deep learning techniques to classify COVID-19 cases from medical images. It explores data handling, model implementation, and evaluation.
- **Dataset:** [COVID-19 Image Dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)
- **Key Topics:**
  - Medical image dataset preprocessing
  - CNN-based classification
  - Training, validation, and testing procedures
  - Performance metrics and visualization

### 6. Optimizers in Deep Learning
- **File:** optimizers.ipynb
- **Description:** An in-depth analysis of various optimization algorithms used in deep learning, such as SGD, Adam, RMSprop, and others. The notebook includes comparisons and performance benchmarks.
- **Key Topics:**
  - Gradient descent variations
  - Comparison of optimizers
  - Impact on model training and convergence
  - Experimental results with different optimizers

### 7. Recurrent Neural Networks (RNN) Model
- **File:** rnn-model.ipynb
- **Description:** A notebook implementing a basic Recurrent Neural Network (RNN) for sequence-based tasks, covering data preparation, model construction, and evaluation.
- **Key Topics:**
  - Understanding RNN structure
  - Time-series and sequential data handling
  - Model training and performance analysis

### 8. Advanced RNN with LSTM
- **File:** cont. rnn-model + LSTM.ipynb
- **Description:** A continuation of the RNN model, incorporating Long Short-Term Memory (LSTM) networks to improve sequence modeling capabilities.
- **Key Topics:**
  - LSTM architecture and its advantages
  - Handling vanishing gradients
  - Training and evaluation on sequential data

### 9. RNN Variants: LSTM & GRU
- **File:** rnn-lstm-gru (1).ipynb
- **Description:** This notebook explores different variations of recurrent neural networks, including LSTMs and Gated Recurrent Units (GRUs), comparing their performance.
- **Key Topics:**
  - LSTM vs. GRU comparison
  - Sequence data handling
  - Performance evaluation and benchmarks

### 10. Regularization Techniques in Deep Learning
- **File:** regulariztion.ipynb
- **Description:** A deep dive into various regularization techniques used to prevent overfitting in deep learning models.
- **Key Topics:**
  - L1 and L2 regularization
  - Dropout and Batch Normalization
  - Effect of regularization on model performance

### 11. Transfer Learning for Image Classification
- **File:** transfer-learnig.ipynb
- **Description:** This notebook explores transfer learning techniques using pre-trained models for image classification. The dataset used is the **Intel Image Classification Dataset**.
- **Dataset:** [Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- **Key Topics:**
  - Understanding transfer learning
  - Using pre-trained models (e.g., VGG16, ResNet, MobileNet)
  - Fine-tuning and feature extraction
  - Performance evaluation and results

### 12. Facial Feature Extraction
- **Dataset:** [Facial Feature Extraction Dataset](https://www.kaggle.com/datasets/osmankagankurnaz/facial-feature-extraction-dataset)
- **Description:** This dataset can be used for various facial recognition and feature extraction tasks, such as expression classification, identity recognition, and biometric security applications.

---

## Installation & Requirements
To run the Jupyter Notebooks, ensure you have the following dependencies installed:

bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow keras jupyter


---

## Usage
1. Clone the repository:
   
bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name

2. Open Jupyter Notebook:
   
bash
   jupyter notebook

3. Navigate to the desired notebook and execute the cells.

---

## Author
**Momen Mohammed Bhais**  
📧 Email: [momenbhais@outlook.com](mailto:momenbhais@outlook.com)

---

## Next Steps
Additional sections will be added covering more advanced deep learning architectures, feature engineering techniques, and real-world datasets.

Stay tuned for updates! 🚀
