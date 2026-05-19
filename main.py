import csv
import numpy as np

def load_csv(filename):
    X, y = [], []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
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

def cost_function(X, y, w, b):
    m = X.shape[0]
    predictions = np.dot(X, w) + b
    cost = np.sum((predictions - y) ** 2) / (2 * m)
    return cost

def gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        err = (np.dot(w, X[i]) + b) - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i, j]
        dj_db += err
    return dj_dw / m, dj_db / m

def gradient_descent(w, b, alpha, dj_dw, dj_db):
    w = w - alpha * dj_dw
    b = b - alpha * dj_db
    return w, b

# Load data
X, y = load_csv('Student_Performance.csv')

# Normalize
mean, std = X.mean(axis=0), X.std(axis=0)
X = (X - mean) / std

# Init
w = np.zeros(X.shape[1])
b = 0
alpha = 0.01
epochs = 1000

# Train
for epoch in range(epochs):
    dj_dw, dj_db = gradient(X, y, w, b)
    w, b = gradient_descent(w, b, alpha, dj_dw, dj_db)
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Cost: {cost_function(X, y, w, b):.4f}")