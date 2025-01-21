Predicting Apple Stock Price Using SARIMAX
This project implements a SARIMAX model to predict the stock prices of Apple Inc. by leveraging historical stock data. The workflow encompasses data preprocessing, feature engineering, model training, and evaluation, following a systematic and reproducible approach.

Table of Contents
Overview
Features
Technologies Used
Installation
Data Preparation
Methodology
Results
Contributions
License
Overview
Stock price prediction is a complex task due to the influence of numerous external factors. This project uses SARIMAX (Seasonal Autoregressive Integrated Moving Average with Exogenous Regressors) to predict future stock prices based on historical data and engineered features.

Features
Automatic Hyperparameter Tuning: Leverages the auto_arima function to identify the best SARIMA parameters.
Seasonal Decomposition: Visualizes the trend, seasonality, and residuals in stock data.
Feature Scaling: Scales features and target variables for improved model convergence.
Performance Evaluation: Provides MSE and RMSE to assess the model's predictive performance.
Technologies Used
Python 3.x
Libraries:
pandas
numpy
matplotlib
scipy
pmdarima
statsmodels
sklearn
Installation
To run this project locally:

Clone the repository:
bash
Copy
Edit
git clone https://github.com/username/apple-stock-sarimax.git
Navigate to the project directory:
bash
Copy
Edit
cd apple-stock-sarimax
Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
Data Preparation
Dataset: Historical stock prices of Apple Inc. should be saved in a CSV file named price_apple.csv. The data must include the following columns:
Date
Low
High
Close
Adj Close
Volume
Loading Data: The script automatically loads the data and preprocesses it, including:
Converting dates to a datetime format.
Engineering a mean column as the average of daily Low and High.
Shifting the mean column to create a target variable (Actual).
Methodology
The workflow is structured as follows:

Exploratory Data Analysis:

Visualize the mean column for trends and seasonality.
Perform seasonal decomposition.
Feature Scaling:

Normalize features and target variables using MinMaxScaler.
Data Splitting:

Split the dataset into training (70%) and testing (30%).
Model Selection:

Automatically tune SARIMA parameters using auto_arima.
Model Training:

Train the SARIMAX model using identified parameters and training data.
Prediction:

Predict future stock prices using the testing set.
Evaluation:

Compute MSE and RMSE for performance assessment.
Results
Model Performance
Mean Squared Error (MSE): 0.00015083
Root Mean Squared Error (RMSE): 0.01228
These metrics indicate that the SARIMAX model predicts Apple's stock price with minimal error.

Visualization
A comparison of actual versus predicted stock prices highlights the model's accuracy:

Contributions
Contributions are welcome! If you encounter any issues or have suggestions for improvements, feel free to submit a pull request or open an issue.
