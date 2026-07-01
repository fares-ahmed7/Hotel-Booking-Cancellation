# 🏨 Hotel Booking Cancellation Prediction System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-REST%20API-black?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipeline-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-blueviolet)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

An end-to-end Machine Learning project that predicts whether a hotel booking will be canceled before check-in.

The project covers the complete ML workflow: data preprocessing, model comparison via cross-validation, model evaluation, deployment with Flask, and containerization with Docker.

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Installation](#️-installation)
- [Training](#-training)
- [Models Compared](#-models-compared)
- [Evaluation Metrics](#-evaluation-metrics)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Model Evaluation](#-model-evaluation)
- [Flask API](#-flask-api)
- [Docker](#-docker)
- [Future Improvements](#-future-improvements)
- [License](#-license)
- [Author](#-author)

---

## 📌 Project Overview

Hotel booking cancellations can significantly impact hotel revenue and operational planning. This project builds several machine learning models to predict booking cancellations and automatically selects the best-performing model based on ROC-AUC cross-validation.

The final solution is deployed as a REST API using Flask and containerized with Docker for easy, reproducible deployment.

---

## ✨ Features

- End-to-end machine learning pipeline
- Automatic data preprocessing (missing value imputation, one-hot encoding, feature scaling)
- Multiple model comparison with 5-fold cross-validation
- Automatic best model selection based on ROC-AUC
- Saved, ready-to-use trained pipeline (`.pkl`)
- Flask REST API for real-time predictions
- Docker support for containerized deployment
- Exploratory Data Analysis (EDA) notebook
- Model evaluation notebook with detailed metrics and plots

---

## 📂 Dataset

**Dataset:** Hotel Booking Demand Dataset

**Target variable:** `is_canceled`

The dataset contains hotel reservation information such as:

- Hotel type
- Lead time
- Arrival date
- Country
- Market segment
- Deposit type
- Customer type
- ADR (Average Daily Rate)
- Previous cancellations
- Special requests
- and more...

---

## 📁 Project Structure

```
Hotel-Booking-Cancellation-Prediction-System/
│
├── app.py                     # Flask REST API
├── train.py                   # Model training script
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
│   └── best_pipeline.pkl      # Trained pipeline (saved after training)
│
├── notebooks/
│   ├── EDA.ipynb               # Exploratory Data Analysis
│   └── Model_evaluation.ipynb  # Model evaluation & comparison
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

## 🛠 Technologies Used

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Machine Learning | Scikit-Learn, XGBoost, LightGBM |
| Visualization | Matplotlib |
| API | Flask, Flask-CORS |
| Model Persistence | Joblib |
| Deployment | Docker |

---

## ⚙️ Installation

**1. Clone the repository**

```bash
git clone https://github.com/fares-ahmed7/Hotel-Booking-Cancellation-Prediction-System.git
```

**2. Move into the project directory**

```bash
cd Hotel-Booking-Cancellation-Prediction-System
```

**3. (Optional) Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

**4. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🚀 Training

Run:

```bash
python train.py
```

The training script automatically:

1. Loads the dataset
2. Splits it into train/test sets
3. Performs preprocessing inside a scikit-learn `Pipeline`
4. Compares multiple models using 5-fold cross-validation
5. Selects the best-performing model
6. Trains the final pipeline
7. Evaluates it on the test set
8. Saves the trained pipeline

**Saved model:**
```
models/best_pipeline.pkl
```

**Saved results:**
```
results/cv_results.csv
results/best_model.txt
```

---

## 🤖 Models Compared

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

The best model is selected automatically based on cross-validated ROC-AUC score.

---

## 📊 Evaluation Metrics

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

## 📈 Exploratory Data Analysis

The EDA notebook (`notebooks/EDA.ipynb`) includes:

- Dataset overview
- Missing values analysis
- Numerical feature distributions
- Categorical feature distributions
- Correlation heatmap
- Outlier detection
- Target distribution
- Feature relationships

---

## 📉 Model Evaluation

The evaluation notebook (`notebooks/Model_evaluation.ipynb`) includes:

- Cross-validation results
- Classification report
- Confusion matrix
- ROC curve
- Performance comparison across models
- Feature importance (where applicable)

---

## 🌐 Flask API

Run the API:

```bash
python app.py
```

Default server:
```
http://localhost:8080
```

### Health Check

**GET** `/health`

```bash
curl http://localhost:8080/health
```

Response:
```json
{
    "status": "healthy (sklearn pipeline)"
}
```

### Prediction

**POST** `/predict`

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

Example response:
```json
{
    "prediction": 0,
    "probability": 0.021
}
```

> `prediction`: `0` = booking not canceled, `1` = booking canceled
> `probability`: model's confidence score for cancellation

---

## 🐳 Docker

**Build the image**

```bash
docker build -t hotel-booking-api .
```

**Run the container**

```bash
docker run -p 8080:8080 hotel-booking-api
```

The API will be available at `http://localhost:8080`.

---

## 📜 License

This project is licensed under the **Apache-2.0 License**. See the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Fares Ahmed**
GitHub: [@fares-ahmed7](https://github.com/fares-ahmed7)

---

⭐ If you found this project useful, consider giving it a star!
