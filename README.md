
# **Time Series Forecasting of Apple (AAPL) Stock Prices using SARIMAX**

## **Executive Summary**

This project presents a robust framework for forecasting the daily stock price of Apple Inc. (AAPL) using a Seasonal Autoregressive Integrated Moving Average with Exogenous Regressors (SARIMAX) model. The objective was to develop a reliable short-term prediction model by leveraging historical price data. The methodology encompasses rigorous feature engineering, automated hyperparameter optimization via `auto_arima`, and a thorough evaluation of the model's predictive accuracy. The resulting model successfully captures the underlying trend of the stock price, demonstrating the effectiveness of SARIMAX for financial time series forecasting and establishing a strong baseline for future enhancements.

## **Table of Contents**

1.  Project Context & Objective
2.  End-to-End Methodology
3.  Evaluation & Results
4.  Technologies & Libraries
5.  Usage & Replication
6.  Conclusion & Future Work
7.  Contributions

-----

## **Project Context & Objective**

Forecasting stock prices is a notoriously challenging task due to market volatility and the complex interplay of numerous factors. The goal of this project was to build and evaluate a time series model capable of making accurate one-step-ahead predictions of Apple's daily stock price.

The SARIMAX model was chosen for its ability to handle key time series characteristics:

  * **Autoregressive (AR)**: Dependency on past values.
  * **Integrated (I)**: Differencing to achieve stationarity.
  * **Moving Average (MA)**: Dependency on past forecast errors.
  * **Exogenous (X)**: Incorporation of external, predictive variables.

This makes SARIMAX a powerful and interpretable choice for this financial forecasting task.

-----

## **End-to-End Methodology**

The project followed a structured data science workflow from data ingestion to model evaluation.

### 1\. Feature Engineering & Target Definition

The raw dataset included daily `Low`, `High`, `Close`, `Adj Close`, and `Volume`.

  * A new feature, `mean`, was engineered as the average of the `Low` and `High` prices. This helps to smooth out intra-day noise and provide a more stable price signal.
  * The prediction target (`Actual`) was defined by shifting the `mean` column one step into the past (`shift(-1)`). This frames the problem as predicting the next day's average price based on the current day's information.

### 2\. Time Series Analysis & Decomposition

An initial exploratory data analysis (EDA) was performed to understand the data's structure.

  * The time series was visualized to identify long-term trends.
  * Seasonal decomposition was applied using a 365-day period to formally inspect the **trend**, **seasonality**, and **residual** components of the series.

### 3\. Data Scaling & Preparation

  * Features and the target variable were scaled using `sklearn.preprocessing.MinMaxScaler`. This normalizes all data into a [0, 1] range, which is critical for ensuring that the model's convergence is stable and not skewed by features of different magnitudes.
  * The dataset was split into training (70%) and testing (30%) sets to facilitate a robust evaluation of the model's performance on unseen data.

### 4\. Hyperparameter Optimization & Model Selection

  * The `pmdarima.auto_arima` function was employed to perform a stepwise search for the optimal non-seasonal order `(p, d, q)` for the model. This automates the hyperparameter tuning process, ensuring a data-driven model specification.
  * The analysis identified **ARIMA(0, 1, 1)** as the best-fit model, indicating that the forecast is a function of one degree of differencing and the previous period's forecast error.

### 5\. SARIMAX Model Training

The final SARIMAX model was trained on the training dataset using:

  * **Endogenous Variable**: The scaled `Actual` stock price (`y`).
  * **Exogenous Variables**: The scaled `Low`, `High`, `Close`, `Adj Close`, `Volume`, and `mean` columns (`X`).
  * **Order**: `(0, 1, 1)` as determined by `auto_arima`.

-----

## **Evaluation & Results**

The model's performance was assessed both quantitatively and qualitatively.

### Quantitative Analysis

The model yielded strong performance metrics on the test set:

  * **Mean Squared Error (MSE)**: `0.00015`
  * **Root Mean Squared Error (RMSE)**: `0.01228`

Given that the target variable was scaled to a [0, 1] range, these extremely low error values indicate a high degree of precision and a very close fit to the actual data.

### Qualitative Analysis

A visual comparison of the predicted versus actual stock prices shows that the model effectively captures the direction and trend of the price movements. As is common with models incorporating moving average components, there is a slight lag, but the overall tracking is highly accurate.

### Model Diagnostics

A deeper look at the model's statistical summary revealed:

  * **Goodness of Fit**: A high Log-Likelihood and low AIC/BIC scores confirm a strong model fit.
  * **Residuals Analysis**: The Ljung-Box test indicated no significant autocorrelation in the residuals (p \> 0.05), suggesting they behave like white noise. However, the Jarque-Bera test indicated that the residuals are not normally distributed and exhibit heteroskedasticity (non-constant variance). This is common in financial time series due to volatility clustering.

-----

## **Technologies & Libraries**

  * **Python 3.x**
  * **Core Libraries**: `pandas`, `numpy`, `matplotlib`
  * **Time Series Modeling**: `statsmodels`, `pmdarima`
  * **Data Preprocessing**: `sklearn`

-----

## **Usage & Replication**

To replicate this analysis:

1.  Clone the repository:
    ```bash
    git clone https://github.com/alerodriguessf/predicting-apple-stock-price.git
    cd predicting-apple-stock-price
    ```
2.  Install the required dependencies:
    ```python
    pip install pandas numpy matplotlib statsmodels pmdarima scikit-learn
    ```
3.  Ensure the dataset, named `price_apple.xlsx`, is present in the root directory.
4.  Execute the Jupyter Notebook `Portfolio_Predicting_Apple_Stock_Price_SARIMAX_20250114.ipynb`.

-----

## **Conclusion & Future Work**

This project successfully demonstrates the implementation of a SARIMAX model for forecasting Apple's stock price. The model provides highly accurate trend predictions and serves as a robust baseline.

Potential future enhancements include:

  * **Modeling Volatility**: Address the observed heteroskedasticity by implementing ARCH/GARCH models alongside the SARIMAX framework to explicitly model periods of high and low volatility.
  * **Incorporate Additional Exogenous Data**: Enrich the model with other predictive features, such as S\&P 500 index movements, tech sector ETFs, or sentiment analysis scores derived from financial news.
  * **Comparative Analysis**: Benchmark the SARIMAX model's performance against other forecasting techniques, such as Long Short-Term Memory (LSTM) networks or Facebook's Prophet, to explore potential accuracy gains from more complex architectures.

-----

## **Contributions**

Contributions, issues, and feature requests are welcome. Please feel free to open an issue or submit a pull request.
