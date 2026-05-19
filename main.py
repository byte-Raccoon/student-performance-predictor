import csv
import numpy as np

def load_csv(filename):
    X, y = [], []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Encode Yes/No to 1/0
            extra = 1 if row['Extracurricular Activities'] == 'Yes' else 0
            features = [
                float(row['Hours Studied']),
                float(row['Previous Scores']),
                extra,
                float(row['Sleep Hours']),
                float(row['Sample Question Papers Practiced'])
            ]
            X.append(features)
            y.append(float(row['Performance Index']))
    return np.array(X), np.array(y)

X, y = load_csv('linear_reg/Student_Performance.csv')

def train_test_split(X, y, test_size=0.2, seed=42):
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    split = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = train_test_split(X, y)

def normalize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0)
    return (X_train - mean) / std, (X_test - mean) / std, mean, std

X_train, X_test, mean, std = normalize(X_train, X_test)

class LinearRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            y_pred = X @ self.weights + self.bias  # dot product

            # Gradients
            dw = (1 / n_samples) * X.T @ (y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update
            self.weights -= self.lr * dw
            self.bias    -= self.lr * db

            if epoch % 100 == 0:
                loss = np.mean((y_pred - y) ** 2)
                print(f"Epoch {epoch} | MSE Loss: {loss:.4f}")

    def predict(self, X):
        return X @ self.weights + self.bias
    
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1 - (ss_res / ss_tot)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Train
model = LinearRegression(lr=0.01, epochs=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"\nR² Score : {r2_score(y_test, y_pred):.4f}")
print(f"MSE      : {mse(y_test, y_pred):.4f}")

new_student = np.array([[6, 85, 1, 7, 3]], dtype=float)
new_student_norm = (new_student - mean) / std  # use same mean/std from training

prediction = model.predict(new_student_norm)
print(f"\nPredicted Performance Index: {prediction[0]:.2f}")