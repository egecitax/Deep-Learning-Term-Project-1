import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
import torch.optim as optim

# 📌 1️⃣ Veri Yükleme ve Ön İşleme
df = pd.read_csv("C:\\Users\\ATC\\Desktop\\BankNote_Authentication.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X, y = df.iloc[:, :-1], df.iloc[:, -1]
scaler = StandardScaler()
X = scaler.fit_transform(X)  # 📌 Veriyi normalize ettik.
y = y.to_numpy().reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 📌 2️⃣ Aktivasyon Fonksiyonları
def sigmoid(Z):
    Z = np.clip(Z, -500, 500)  # Sayısal taşmayı önlemek için değerleri sınırlıyoruz
    return 1 / (1 + np.exp(-Z))


def relu(Z):
    return np.maximum(0, Z)

def tanh_derivative(A):
    A = np.clip(A, -1, 1)  # `A`'yı -1 ile 1 arasında sınırla
    return 1 - np.square(A)



# 📌 3️⃣ Sinir Ağı Modeli (2-Layer ve 3-Layer)
class NeuralNetwork:
    def __init__(self, n_x, n_h1, n_h2=None, n_y=1, activation="relu", learning_rate=0.003):
        np.random.seed(42)
        self.learning_rate = learning_rate
        self.activation = activation
        self.parameters = {
            "W1": np.random.randn(n_h1, n_x) * np.sqrt(1. / n_x),
            "b1": np.zeros((n_h1, 1))
        }
        if n_h2:
            self.parameters.update({
                "W2": np.random.randn(n_h2, n_h1) * np.sqrt(1. / n_h1),
                "b2": np.zeros((n_h2, 1)),
                "W3": np.random.randn(n_y, n_h2) * np.sqrt(1. / n_h2),
                "b3": np.zeros((n_y, 1))
            })
        else:
            self.parameters.update({
                "W2": np.random.randn(n_y, n_h1) * np.sqrt(1. / n_h1),
                "b2": np.zeros((n_y, 1))
            })

    def forward_propagation(self, X):
        W1, b1 = self.parameters["W1"], self.parameters["b1"]
        W2, b2 = self.parameters["W2"], self.parameters["b2"]

        Z1 = np.dot(W1, X.T) + b1
        A1 = np.tanh(Z1) if self.activation == "tanh" else relu(Z1)

        if "W3" in self.parameters:
            W3, b3 = self.parameters["W3"], self.parameters["b3"]
            Z2 = np.dot(W2, A1) + b2
            A2 = np.tanh(Z2) if self.activation == "tanh" else relu(Z2)
            Z3 = np.dot(W3, A2) + b3
            A3 = sigmoid(Z3)
            return A3.T, {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3}

        Z2 = np.dot(W2, A1) + b2
        A2 = sigmoid(Z2)
        return A2.T, {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

    def back_propagation(self, X, Y, cache):
        m = X.shape[0]
        grads = {}

        A1 = cache["A1"]
        Z1 = cache["Z1"]

        if "A3" in cache:  # 3-Layer modeli kontrol et
            A2, A3 = cache["A2"], cache["A3"]
            dZ3 = A3 - Y.T
            grads["dW3"] = (1 / m) * np.dot(dZ3, A2.T)
            grads["db3"] = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

            dA2 = np.dot(self.parameters["W3"].T, dZ3)
            dZ2 = dA2 * tanh_derivative(A2)
            grads["dW2"] = (1 / m) * np.dot(dZ2, A1.T)
            grads["db2"] = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

            dA1 = np.dot(self.parameters["W2"].T, dZ2)
            dZ1 = dA1 * tanh_derivative(A1)
            grads["dW1"] = (1 / m) * np.dot(dZ1, X)
            grads["db1"] = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        else:  # 2-Layer modeli için
            A2 = cache["A2"]
            dZ2 = A2 - Y.T
            grads["dW2"] = (1 / m) * np.dot(dZ2, A1.T)
            grads["db2"] = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

            dA1 = np.dot(self.parameters["W2"].T, dZ2)
            dZ1 = dA1 * tanh_derivative(A1)
            grads["dW1"] = (1 / m) * np.dot(dZ1, X)
            grads["db1"] = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        return grads

    def compute_cost(self, A_last, Y):
        m = Y.shape[0]
        epsilon = 1e-8  # Küçük bir değer ekleyerek log(0) hatasını önleyelim
        cost = -(np.dot(Y.T, np.log(A_last + epsilon)) + np.dot((1 - Y).T, np.log(1 - A_last + epsilon))) / m
        return float(np.squeeze(cost))


    def update_parameters(self, grads):
        for key in self.parameters.keys():
            if "d" + key in grads:
                self.parameters[key] -= self.learning_rate * grads["d" + key]

    def train(self, X, Y, n_steps=5000):
        for i in range(n_steps):
            A_last, cache = self.forward_propagation(X)
            cost = self.compute_cost(A_last, Y)

            grads = self.back_propagation(X, Y, cache)  # Backpropagation eklendi
            self.update_parameters(grads)  # Ağırlık güncellemesi yapıldı

            if i % 1000 == 0:
                print(f"Step {i}, Cost: {cost:.4f}")

    def predict(self, X):
        A_last, _ = self.forward_propagation(X)
        return (A_last > 0.5).astype(int)

class PyTorchNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PyTorchNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.Tanh()  # ReLU versiyon için değiştirilebilir
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

input_size = X_train.shape[1]
hidden_size = 6
output_size = 1

pytorch_model = PyTorchNN(input_size, hidden_size, output_size)

# Kayıp fonksiyonu ve optimizasyon
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.SGD(pytorch_model.parameters(), lr=0.003)

X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.Tensor(y_train)
X_test_tensor = torch.Tensor(X_test)
y_test_tensor = torch.Tensor(y_test)

num_epochs = 5000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = pytorch_model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

# PyTorch Modeli Test Sonuçları
with torch.no_grad():
    y_pred_pytorch = pytorch_model(X_test_tensor)
    y_pred_pytorch = (y_pred_pytorch.numpy() > 0.5).astype(int)  # 0.5 eşik değeri

# PyTorch Confusion Matrix & Classification Report
print("\n🔹 PyTorch Modeli İçin Sonuçlar 🔹")
print(f"PyTorch Accuracy: {accuracy_score(y_test, y_pred_pytorch):.4f}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_pytorch))
print("\nClassification Report:\n", classification_report(y_test, y_pred_pytorch))




# 📌 2-Layer ve 3-Layer Model Eğitme(tanh)
nn_2L_tanh = NeuralNetwork(X_train.shape[1], 6, activation="tanh")
nn_2L_tanh.train(X_train, y_train, n_steps=5000)

nn_3L_tanh = NeuralNetwork(X_train.shape[1], 6, 6, activation="tanh")
nn_3L_tanh.train(X_train, y_train, n_steps=5000)

y_pred_2L_tanh = nn_2L_tanh.predict(X_test)
y_pred_3L_tanh = nn_3L_tanh.predict(X_test)

print("Confusion Matrix(2-L-tanh):\n", confusion_matrix(y_test, y_pred_2L_tanh))
print("\nClassification Report(2-L-tanh):\n", classification_report(y_test, y_pred_2L_tanh))

print("Confusion Matrix(3-L-tanh):\n", confusion_matrix(y_test, y_pred_3L_tanh))
print("\nClassification Report(3-L-tanh):\n", classification_report(y_test, y_pred_3L_tanh))

print(f"2-Layer-tanh Accuracy: {accuracy_score(y_test, y_pred_2L_tanh):.4f}")
print(f"3-Layer-tanh Accuracy: {accuracy_score(y_test, y_pred_3L_tanh):.4f}")

# 📌 Scikit-learn Modeli(tanh)
mlp = MLPClassifier(hidden_layer_sizes=(6, 6), activation='tanh', solver='sgd', max_iter=5000)
mlp.fit(X_train, y_train.ravel())

y_pred_sklearn_tanh = mlp.predict(X_test)

print("Scikit-learn-tanh Accuracy:", mlp.score(X_test, y_test))
print("Confusion Matrix (sklearn_tanh):\n", confusion_matrix(y_test, y_pred_sklearn_tanh))
print("\nClassification Report(sklearn_tanh):\n", classification_report(y_test, y_pred_sklearn_tanh))

# 📌 2-Layer ve 3-Layer Model Eğitme(ReLu)
nn_2L_relu = NeuralNetwork(X_train.shape[1], 6, activation="relu")
nn_2L_relu.train(X_train, y_train, n_steps=5000)

nn_3L_relu = NeuralNetwork(X_train.shape[1], 6, 6, activation="relu")
nn_3L_relu.train(X_train, y_train, n_steps=5000)

y_pred_2L_relu = nn_2L_relu.predict(X_test)
y_pred_3L_relu = nn_3L_relu.predict(X_test)

print("Confusion Matrix(2-L-relu):\n", confusion_matrix(y_test, y_pred_2L_relu))
print("\nClassification Report(2-L-relu):\n", classification_report(y_test, y_pred_2L_relu))

print("Confusion Matrix(3-L-relu):\n", confusion_matrix(y_test, y_pred_3L_relu))
print("\nClassification Report(3-L-relu):\n", classification_report(y_test, y_pred_3L_relu))

print(f"2-Layer-relu Accuracy: {accuracy_score(y_test, y_pred_2L_relu):.4f}")
print(f"3-Layer-relu Accuracy: {accuracy_score(y_test, y_pred_3L_relu):.4f}")

# 📌 Scikit-learn Modeli(ReLu)
mlp1 = MLPClassifier(hidden_layer_sizes=(6, 6), activation='relu', solver='sgd', max_iter=5000)
mlp1.fit(X_train, y_train.ravel())

y_pred_sklearn_relu = mlp1.predict(X_test)

print("Scikit-learn-relu Accuracy:", mlp1.score(X_test, y_test))
print("Confusion Matrix (sklearn_relu):\n", confusion_matrix(y_test, y_pred_sklearn_relu))
print("\nClassification Report(sklearn_relu):\n", classification_report(y_test, y_pred_sklearn_relu))


models = {
    "2L_tanh": (nn_2L_tanh, accuracy_score(y_test, nn_2L_tanh.predict(X_test))),
    "3L_tanh": (nn_3L_tanh, accuracy_score(y_test, nn_3L_tanh.predict(X_test))),
    "2L_relu": (nn_2L_relu, accuracy_score(y_test, nn_2L_relu.predict(X_test))),
    "3L_relu": (nn_3L_relu, accuracy_score(y_test, nn_3L_relu.predict(X_test))),
    "scikit_tanh": (mlp, accuracy_score(y_test, y_pred_sklearn_tanh)),
    "scikit_relu": (mlp1, accuracy_score(y_test, y_pred_sklearn_relu)),
    "pytorch": (pytorch_model, accuracy_score(y_test, y_pred_pytorch)),
}

# En yüksek doğruluk oranına sahip modeli seçelim
best_model = max(models.items(), key=lambda x: x[1][1])
print(f"\n🔥 Seçilen En İyi Model: {best_model[0]}, Accuracy: {best_model[1][1]:.4f}")

