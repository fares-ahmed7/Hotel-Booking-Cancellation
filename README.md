# 🏨 Hotel Booking Cancellation Prediction

An end-to-end Machine Learning project that predicts whether a hotel booking will be canceled before check-in.

The project covers the complete ML workflow, including data preprocessing, model comparison using cross-validation, model evaluation, deployment with Flask, and containerization using Docker.

---

# 📌 Project Overview

Hotel booking cancellations can significantly impact hotel revenue and operational planning. This project builds several machine learning models to predict booking cancellations and automatically selects the best-performing model based on ROC-AUC cross-validation.

The final solution is deployed as a REST API using Flask and containerized with Docker.

---

# ✨ Features

- End-to-End Machine Learning Pipeline
- Automatic Data Preprocessing
- Missing Value Imputation
- One-Hot Encoding
- Feature Scaling
- Multiple Model Comparison
- 5-Fold Cross Validation
- Automatic Best Model Selection
- Model Evaluation
- Saved Trained Pipeline
- Flask REST API
- Docker Support
- Exploratory Data Analysis (EDA)
- Model Evaluation Notebook

---

# 📂 Dataset

Dataset:
Hotel Booking Demand Dataset

Target Variable:

```
is_canceled
```

Dataset contains hotel reservation information such as:

- Hotel Type
- Lead Time
- Arrival Date
- Country
- Market Segment
- Deposit Type
- Customer Type
- ADR
- Previous Cancellations
- Special Requests
- and many more...

---

# 📁 Project Structure

```text
Hotel-Booking-Cancellation/
│
├── app.py
├── train.py
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── LICENSE
├── README.md
│
├── data/
│   └── hotel_bookings.csv
│
├── models/
│   └── best_pipeline.pkl
│
├── notebooks/
│   ├── EDA.ipynb
│   └── Model_evaluation.ipynb
│
├── results/
│   ├── best_model.txt
│   └── cv_results.csv
│
└── src/
    ├── pipeline.py
    ├── preprocessing.py
    └── evaluation.py
```

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- LightGBM
- Matplotlib
- Flask
- Flask-CORS
- Joblib
- Docker

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/fares-ahmed7/Hotel-Booking-Cancellation.git
```

Move into the project directory

```bash
cd Hotel-Booking-Cancellation
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Training

Run

```bash
python train.py
```

The training script automatically:

- Loads the dataset
- Splits train/test data
- Performs preprocessing inside a Scikit-learn Pipeline
- Compares multiple models using 5-Fold Cross Validation
- Selects the best model
- Trains the best pipeline
- Evaluates on the test set
- Saves the trained pipeline

Saved model:

```
models/best_pipeline.pkl
```

Saved results:

```
results/cv_results.csv
results/best_model.txt
```

---

# 🤖 Models Compared

The following models are evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

The best model is selected automatically based on Cross-Validation ROC-AUC.

---

# 📊 Evaluation Metrics

The final model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Classification Report
- Confusion Matrix
- ROC Curve

---

# 📈 Exploratory Data Analysis

The EDA notebook includes:

- Dataset Overview
- Missing Values Analysis
- Numerical Feature Distributions
- Categorical Feature Distributions
- Correlation Heatmap
- Outlier Detection
- Target Distribution
- Feature Relationships

Notebook:

```
notebooks/01_eda.ipynb
```

---

# 📉 Model Evaluation

The evaluation notebook includes:

- Cross Validation Results
- Classification Report
- Confusion Matrix
- ROC Curve
- Performance Comparison
- Feature Importance (where applicable)

Notebook:

```
notebooks/02_model_evaluation.ipynb
```

---

# 🌐 Flask API

Run the API

```bash
python app.py
```

Default server:

```
http://localhost:8080
```

---

## Health Check

**GET**

```
/health
```

Response

```json
{
    "status": "healthy (sklearn pipeline)"
}
```

---

## Prediction

**POST**

```
/predict
```

Example Request

```json
{
    "hotel": "Resort Hotel",
    "lead_time": 342,
    "arrival_date_year": 2015,
    "arrival_date_month": "July",
    "arrival_date_week_number": 27,
    "arrival_date_day_of_month": 1,
    "stays_in_weekend_nights": 0,
    "stays_in_week_nights": 0,
    "adults": 2,
    "children": 0,
    "babies": 0,
    "meal": "BB",
    "country": "PRT",
    "market_segment": "Direct",
    "distribution_channel": "Direct",
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "reserved_room_type": "C",
    "assigned_room_type": "C",
    "booking_changes": 3,
    "deposit_type": "No Deposit",
    "agent": 0,
    "days_in_waiting_list": 0,
    "customer_type": "Transient",
    "adr": 0,
    "required_car_parking_spaces": 0,
    "total_of_special_requests": 0
}
```

Example Response

```json
{
    "prediction": 0,
    "probability": 0.021
}
```

---

# 🐳 Docker

Build Docker Image

```bash
docker build -t hotel-booking-api .
```

Run Container

```bash
docker run -p 8080:8080 hotel-booking-api
```

---

# 📌 Future Improvements

- Hyperparameter Optimization
- Feature Selection
- CI/CD Pipeline
- Cloud Deployment
- Model Monitoring
- MLflow Integration
- Streamlit Dashboard

---

# 📜 License

This project is licensed under the MIT License.

See the LICENSE file for details.

---

# 👨‍💻 Author

**Fares Ahmed**

GitHub:
https://github.com/fares-ahmed7

---

# ⭐ If you found this project useful, consider giving it a Star!