# Customer Retention Prediction System

An end-to-end machine learning system for identifying telecom customers at risk of churn and supporting targeted customer retention decisions.

This project goes beyond building a binary classification model. It explores how a churn prediction system can be designed around the **business cost of a targeted customer retention campaign**, then translates the experimentation into a modular, reproducible ML system with preprocessing, model training, evaluation, experiment tracking, API serving, testing, and containerization.

## Table of Contents

* [Problem Statement](#problem-statement)
* [Project Objective](#project-objective)
* [Dataset](#dataset)
* [Approach](#approach)

  * [1. Data Cleaning](#1-data-cleaning)
  * [2. Data Splitting](#2-data-splitting)
  * [3. Data Preprocessing](#3-data-preprocessing)
  * [4. Business-Aware Evaluation](#4-business-aware-evaluation)
  * [5. Model Selection](#5-model-selection)
  * [6. Hyperparameter Tuning](#6-hyperparameter-tuning)
* [Productionization](#productionization)
* [Project Structure](#project-structure)
* [Experiment Tracking](#experiment-tracking)
* [Testing](#testing)
* [API](#api)
* [Docker](#docker)
* [Installation](#installation)
* [Running the Project](#running-the-project)
* [Key Design Decisions](#key-design-decisions)
* [Limitations and Future Improvements](#limitations-and-future-improvements)

## Problem Statement

Customer churn is a major challenge for subscription-based businesses.

When a customer leaves, the business loses the future revenue associated with that customer. A retention team therefore needs to identify customers who are likely to churn before the customer actually leaves, so that targeted retention actions can be taken.

The goal of this project is to build a system that answers:

> **Which customers are most likely to churn, and how should the prediction threshold be chosen when the cost of missing a churner is different from the cost of targeting a loyal customer?**

A conventional classification model may optimize a metric such as accuracy or F1 score. However, for customer retention, the consequences of different types of mistakes are not necessarily equal.

* **False Negative:** A customer who eventually churns is predicted as loyal.
* **False Positive:** A loyal customer is incorrectly identified as likely to churn and may receive an unnecessary retention intervention.

Missing a genuine churner can represent a significantly greater business cost than contacting a customer who would have stayed anyway.

This project therefore treats churn prediction as a **cost-sensitive decision problem** rather than simply a classification problem.

## Project Objective

The system was designed to:

1. Clean and transform raw customer data.
2. Build a reproducible preprocessing pipeline.
3. Establish a Logistic Regression baseline.
4. Compare multiple machine learning algorithms.
5. Define an evaluation metric that reflects the assumed business costs.
6. Tune the prediction threshold based on that business cost.
7. Optimize model hyperparameters using Optuna.
8. Track experiments and model artifacts using MLflow.
9. Separate experimentation from production-oriented code.
10. Expose the trained model through an API.
11. Containerize the application with Docker.
12. Add automated tests around important components of the system.

## Dataset

The project uses the **Telco Customer Churn** dataset.

The dataset contains 7,043 customer records and 21 columns, including customer demographics, account information, subscribed services, billing information, and the churn target.

The dataset contains 5,174 non-churned customers and 1,869 churned customers, making class imbalance an important consideration during modeling.

# Approach

## 1. Data Cleaning

The raw dataset required several preprocessing steps before it could be used for machine learning.

### Removing identifiers

`customerID` was removed because it uniquely identifies customers but does not provide useful predictive information for the model.

Keeping arbitrary identifiers can allow a model to learn patterns that do not generalize to new customers.

### Correcting `TotalCharges`

Although `TotalCharges` represents a numerical quantity, it was initially loaded as an object/string column.

It was converted explicitly to numeric values. Values that cannot be interpreted as numbers are converted into missing values, allowing them to be handled consistently by the preprocessing pipeline.

## 2. Data Splitting

During experimentation, the dataset was split into training, validation, and test sets. The split was stratified to preserve the churn/non-churn distribution across the datasets.

The training set was used to train the model, the validation set was used for model comparison, threshold selection, and hyperparameter tuning, while the test set was kept for final evaluation.

During the final training, the dataset was split into training and test sets. The training set used to train the model, while the test set was used to evaluate it.

## 3. Data Preprocessing

Rather than manually transforming columns before every modeling experiment, preprocessing was organized into reusable pipelines.

### Numerical features

Numerical variables include:

* `tenure`
* `MonthlyCharges`
* `TotalCharges`

The numerical preprocessing pipeline performs:

1. Median imputation
2. Standard scaling

### Categorical features

Categorical variables were treated as nominal features because there was no assumption of a strict ordering between their categories.

The categorical preprocessing pipeline performs:

1. Most-frequent imputation
2. One-hot encoding
3. Ignoring previously unseen categories during inference

These transformations are combined using a `ColumnTransformer`.

This design ensures that the same preprocessing logic can be applied consistently during training and inference.

## 4. Business-Aware Evaluation

One of the central design decisions in this project was to avoid relying exclusively on conventional classification metrics.

### Accuracy isn't enough

Suppose a model predicts that almost every customer will stay.

Because most customers in the dataset do not churn, such a model could achieve reasonable accuracy while failing to identify a large proportion of customers who are actually going to leave.

More importantly, the cost of the mistakes is asymmetric.

For this project, the following business assumption was made:

> **Missing a churner is 6 times more costly than incorrectly targeting a loyal customer.**

This assumption was incorporated directly into a custom evaluation function.

### Business Cost

The base cost is defined as:

```text
Business Cost = (6 × False Negatives) + (1 × False Positives)
```

However, simply optimizing recall would encourage the model to flag too many customers.

A precision bottleneck was therefore introduced. If precision falls below the chosen threshold, an additional penalty is applied.

The implemented function is conceptually:

```text
Base Cost
    +
Precision Penalty
```

The precision penalty increases as precision falls below the specified threshold.

This allows the evaluation metric to reflect two competing business concerns:

* Missing customers who are likely to churn
* Avoiding unnecessary retention interventions

The actual implementation calculates the confusion-matrix components, applies the 6:1 false-negative/false-positive cost ratio, and adds the precision penalty when required.

That makes the evaluation much closer to the eventual decision-making context.

## 5. Prediction Threshold Selection

Classification models typically use a probability threshold of 0.5 to convert predicted probabilities into classes.

For a retention system, there is no reason to assume that 0.5 is automatically the optimal threshold.

A lower threshold can identify more potential churners, increasing recall, but it also increases the number of customers incorrectly flagged.

The validation set was therefore used to examine different thresholds.

For the tuned XGBoost experiment, the validation results around the selected threshold demonstrated the trade-off between precision, recall and business cost. At a threshold of 0.30, the experiment produced a validation cost of 421 with approximately 0.454 precision and 0.882 recall for the churn class.

## 6. Model Selection

Several models were considered rather than immediately choosing a complex algorithm.

### Logistic Regression

Logistic Regression was used as the baseline model.

This establishes a simple reference point against which more complex models can be evaluated.

### Random Forest

Random Forest was introduced as a non-linear ensemble model capable of capturing interactions that a linear model may miss.

At the selected threshold, the Random Forest experiment produced a higher business cost than the baseline experiment.

### LightGBM

LightGBM was evaluated as a gradient-boosting approach.

The initial LightGBM experiment produced a lower validation cost than both the Logistic Regression and Random Forest experiments.

### XGBoost

XGBoost was also evaluated and subsequently tuned.

The initial XGBoost model used `scale_pos_weight` to account for the imbalance between churned and non-churned customers.

The final model-selection process considered not only conventional classification metrics but, importantly, the custom business cost.

This made model selection consistent with the actual objective of the system.

## Hyperparameter Tuning

After comparing the initial models, hyperparameter optimization was performed using Optuna.

For XGBoost, parameters explored included:

* `n_estimators`
* `learning_rate`
* `max_depth`
* `subsample`
* `colsample_bytree`
* `min_child_weight`
* `gamma`
* `reg_alpha`
* `reg_lambda`

The objective function trained an XGBoost model, generated churn probabilities, converted them into predictions using the selected threshold, and returned the custom `business_cost`. 
In other words, the hyperparameter search was explicitly optimized toward the **business objective**, rather than generic accuracy.

This produced the best set of parameters for the XGBoost model.

## Productionization

After the initial modeling work was developed in a notebook, the project was reorganized into a modular machine learning system.

The goal was to separate responsibilities such as:

* Data loading
* Data validation
* Preprocessing
* Model training
* Model evaluation
* API inference

This makes the system easier to maintain and, more importantly, reduces the risk of having training logic that differs from inference logic.

The notebook served as the experimentation and reasoning layer, while the modularized source code contains the reusable implementation.

## Project Structure

```text
Customer-Retention-Prediction-System/
│
├── data/
│   ├── Telco-Customer-Churn-Raw.csv
│   └── Telco-Customer-Churn-Processed.csv
│
├── mlruns/
│
├── models/
│   └── telco_churn_pipeline.pkl
│
├── notebooks/
│   └── Telco_Churn_Experimentation.ipynb
│
├── scripts/
│   └── train.py
│
├── src/
│   ├── api/
|   |   ├── app.py
│   │   └── main.py
│   │
│   ├── data/
│   │   ├── load_data.py
│   │   ├── preprocess_data.py
│   │   ├── save_data.py
│   │   └── validate_data.py
│   │
│   ├── features/
│   │   ├── transform_data.py
│   │   └── transform_pipeline.py
│   │
│   └── models/
│       ├── evaluate.py
│       ├── train.py
│       └── tune.py
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_training.py
│   └── test_api.py
│
├── .dockerignore
├── .gitignore
├── dockerfile
├── README.md
└── requirements.txt
```

The structure separates experimentation from reusable application code while keeping each stage of the ML lifecycle responsible for a specific task.

## Experiment Tracking

MLflow was incorporated to track model experiments.

The tracked information includes:

* Model parameters
* Precision
* Recall
* Business cost
* Trained model artifact

This provides a foundation for comparing experiments without relying solely on manually recorded notebook outputs.

## Testing

The project includes automated tests covering important components of the system.

Tests are intended to verify that:

* Data preprocessing behaves as expected.
* Model training produces a valid trained model.
* The API accepts valid inputs and handles invalid inputs appropriately.

Testing is particularly important in a production-oriented ML project because a model can produce technically valid predictions while the surrounding data transformation or serving logic is incorrect.

## API

The trained model was exposed through a FastAPI application.

The API provides a way for external applications to submit customer information and obtain a churn prediction without interacting directly with the training code. This allows the trained model to be reused without rerunning the experimentation workflow.

## Docker

The application was containerized using Docker.

Containerization packages the application and its Python dependencies into a reproducible environment, reducing the dependency on the configuration of the machine running the application.

This also provides a cleaner path toward deploying the model to a cloud or server environment in the future.

## Installation

Clone the repository:

```bash
git clone https://github.com/korede-ishola/Customer-Retention-Prediction-System.git

cd Customer-Retention-Prediction-System
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Train the model

The training process can be executed through the training script:

```bash
python scripts/train.py
```

### Run the API

Start the FastAPI application with:

```bash
uvicorn src.api.main:app --reload
```

The API documentation is available through FastAPI's automatically generated Swagger interface.

### Run tests

```bash
pytest
```

### Run with Docker

Build the image:

```bash
docker build -t customer-retention-system .
```

Then run the container:

```bash
docker run -p 8000:8000 customer-retention-system
```

## Limitations and Future Improvements

This project represents a production-oriented implementation rather than a fully deployed production system.

Potential future improvements include:

* More extensive threshold optimization based on real retention campaign costs.
* Monitoring for data and model drift.
* Automated retraining.
* Model versioning and promotion workflows.
* Integration with a customer relationship management system.
* Deployment to a cloud environment.

Most importantly, the assumed business costs in this project are illustrative. A real organization should estimate the financial impact of missed churners, retention offers, and different customer segments using historical business data.

## Conclusion

This project demonstrates an end-to-end approach to building a machine learning system for customer retention.

Rather than simply training a model that predicts churn, it explores how churn predictions can be turned into a decision system that reflects the underlying business cost.
