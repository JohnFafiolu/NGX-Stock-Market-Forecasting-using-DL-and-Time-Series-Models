# Hybrid SARIMA-Deep Learning Framework for Nigerian Stock Market Forecasting 📈

## Project Overview
This repository contains the complete codebase for predicting the very volatile Nigerian Exchange Group All-Share Index (NGX ASI). Traditional Time Series Models  often fail to capture sudden market shocks in emerging economies. This project solves that problem by developing a hybrid forecasting framework that utilizes a Seasonal ARIMA (SARIMA) model to map linear trends and a Deep Learning network to model the non-linear market residuals via the error-correction principle.

## Key Objectives
1. Perform diagnostic testing (Stationarity and Seasonality) on NGX ASI historical data.
2. Formulate a standalone SARIMA baseline using walk-forward (rolling) validation.
3. Conduct a tournament among three deep learning architectures (RNN, LSTM, and GRU) to model the non-linear residuals.
4. Evaluate models using five distinct error metrics (MAE, RMSE, MAPE, SMAPE, MASE).
5. Prove that the Hybrid (SARIMA + Neural Network) framework significantly minimizes forecast errors compared to standalone statistical and AI models.

## Methodology
* **Data:** Daily closing prices of the NGX ASI (Jan 2012 – Nov 2025) gotten from Investing.com.
* **Linear Component:** Auto-SARIMA with $d=1$ (differencing) and $m=5$ (weekly seasonality).
* **Non-Linear Component:** Scaled residual sequence modeling using Keras/TensorFlow.
* **Hybrid Formulation:** $Y_t = L_t + N_t$ (Final Forecast = Linear Prediction + Deep Learning Error Correction).

## Results
A deep learning tournament was conducted on the test set residuals. The **RNN** emerged as the champion architecture. 

When applied to the final test set, the Hybrid Framework significantly outperformed the baseline models:
* **Standalone SARIMA MAPE:** 0.4792%
* **Standalone Deep Learning MAPE:** 18.0491%
* **Hybrid Model MAPE:** 0.4767%

*(Include a screenshot of your final Zoomed-In Matplotlib chart here)*

## How to Run This Project
1. Clone this repository to your local machine.
2. Install the required dependencies using:
   `pip install -r requirements.txt`
3. Open `Hybrid Model for NGX ASI.ipynb` in Jupyter Notebook.
4. Run all cells to execute the data processing, SARIMA baseline, DL tournament, and final evaluation.

## Author
* **John Olukayode Fafiolu** - Data Analyst & Statistician
