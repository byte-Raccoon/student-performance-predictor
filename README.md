# 📊 Student Performance Predictor

A linear regression model built from scratch in Python that predicts a student's Performance Index based on study habits and lifestyle factors — using only NumPy and csv, no ML libraries.

---

## Dataset

`Student_Performance.csv` — 10,000 student records with the following features:

| Feature | Description |
|---|---|
| Hours Studied | Daily study hours |
| Previous Scores | Scores from previous exams |
| Extracurricular Activities | Yes / No (encoded to 1 / 0) |
| Sleep Hours | Average sleep per night |
| Sample Question Papers Practiced | Number of practice papers |
| Performance Index | Target variable (0–100) |

---

## How to Run

Make sure you have Python 3 and NumPy installed.

```bash
pip install numpy
python3 main.py
```

---

## How It Works

1. Loads and preprocesses the CSV (Yes/No encoded to 1/0)
2. Normalizes all features using mean and standard deviation
3. Initializes weights and bias to zero
4. Computes gradients manually using batch gradient descent
5. Updates weights and bias each epoch
6. Prints cost every 100 epochs to track training progress

---

## Project Structure

```
main.py                     # Model implementation
Student_Performance.csv     # Dataset
README.md                   # Project documentation
```

---

## Built With

- Python 3
- NumPy
- Inspired by Andrew Ng's Machine Learning Specialization (Course 1)

---

## Author

**Sanket Dalal**  
B.Tech IT — Delhi Technological University  
[GitHub](https://github.com/byte-Raccoon) | [LinkedIn](https://linkedin.com/in/sanket-dalal)
