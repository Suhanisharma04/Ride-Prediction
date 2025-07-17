# Ride-share Price Estimator

A machine learning-powered application that predicts ride-share prices based on trip details such as hour of the day, distance, surge pricing, visibility, cab company, ride type, and source area. Built using Python, scikit-learn, and Gradio for deployment.

---

## Project Overview

This project aims to model and predict the price of ride-share services (like Uber and Lyft) using publicly available trip data. The application demonstrates data cleaning, exploratory data analysis, regression modeling, and interactive deployment—all in one pipeline.

---

## Key Features

- **Linear Regression Model** trained on ride-share trip data.
- **Real-time Price Estimation** using a Gradio interface.
- Handles **missing values, outliers, and categorical variables**.
- Includes **exploratory visualizations** like price distribution and hourly trends.
- Easily deployable and extendable.

---

## Live Demo

You can run this app locally by installing the dependencies and executing the Python script.  
Gradio will generate a shareable link after launching.

---

## Project Summary

This ride-share price prediction system estimates the fare of a trip based on input details such as time, distance, visibility, ride type, and more. It was built using:

- **Python** for backend logic and model development
- **Pandas, Scikit-learn, and Matplotlib** for data cleaning, analysis, and machine learning
- **Gradio** for creating an interactive and user-friendly web interface
- **Pickle** for model persistence


---

## How It Works

1. **Data Cleaning**:
   - Converts price to numeric and removes rows with missing/invalid entries.
   - Fills remaining missing values and removes extreme outliers in price.

2. **Feature Engineering**:
   - One-hot encodes categorical features: cab type, ride type, source area.
   - Selects relevant features for model training.

3. **Model Training**:
   - Uses Linear Regression from scikit-learn.
   - Evaluates model performance using R² and Mean Squared Error.

4. **Model Saving**:
   - Saves the model and its input feature structure using `pickle`.

5. **Deployment**:
   - A Gradio interface allows users to input trip conditions and get a predicted fare.




## Project Status
Completed as part of a data science module; demonstrates practical application of regression models.
