BankNote Authentication using Neural Networks

Introduction

The objective of this project is to classify banknotes as either authentic or counterfeit using machine learning models. The dataset used is "BankNote_Authentication.csv", sourced from Kaggle. The study involves training neural network models with different architectures and activation functions to compare their performance. The final model selection is based on accuracy and computational efficiency.

Methods

Dataset and Preprocessing

The dataset consists of features extracted from genuine and counterfeit banknotes.

Features are standardized using StandardScaler to normalize input values.

The data is split into 80% training and 20% testing using train_test_split().

Model Architectures

We implemented multiple models for comparison:

Custom Neural Networks (Implemented in Python using NumPy):

2-Layer Model (1 hidden layer)

3-Layer Model (2 hidden layers)

Scikit-learn MLP Classifier

PyTorch Feedforward Neural Network

Activation Functions Tested

tanh

ReLU

Loss Function and Optimization

Binary Cross Entropy (BCE) was used as the loss function.

Stochastic Gradient Descent (SGD) was used for optimization.

Models were trained for 5000 epochs.

Evaluation Metrics

Accuracy

Precision, Recall, and F1-score

Confusion Matrix

Results

Performance Comparison

Model

Activation

Accuracy

2-Layer Custom NN

tanh

98.18%

3-Layer Custom NN

tanh

98.55%

2-Layer Custom NN

ReLU

95.27%

3-Layer Custom NN

ReLU

92.00%

Scikit-learn MLP

tanh

99.27%

Scikit-learn MLP

ReLU

99.27%

PyTorch NN

tanh

98.18%

Best Model Selection

The Scikit-learn MLP (tanh) achieved the highest accuracy (99.27%).

Custom models performed well but were slightly outperformed by Scikit-learn.

PyTorch achieved a strong result but was slightly behind.

Discussion

Impact of Activation Functions:

tanh performed well, maintaining stability.

ReLU showed better results in some cases but was less stable.

Comparison of Models:

Custom NN models provided good accuracy but required more training iterations.

Scikit-learn’s MLP was computationally more efficient and provided the best accuracy.

PyTorch model was competitive but slightly behind Scikit-learn in accuracy.

Future Improvements:

Implement dropout to prevent overfitting.

Use Adam optimizer instead of SGD for faster convergence.

Test with different hidden layer sizes and depths.

References

Kaggle Dataset: https://www.kaggle.com/

Andrew Ng’s Deep Learning Course: https://www.deeplearning.ai/

Scikit-learn MLP Documentation: https://scikit-learn.org/

PyTorch Documentation: https://pytorch.org/
