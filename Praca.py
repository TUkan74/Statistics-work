import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams
import pmdarima as pm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error

# Load the data
data = pd.read_csv("tsla.us.txt", index_col="Date", parse_dates=True)
data_close = data["Close"]


# Test for stationarity
def test_stationarity(timeseries):
    # Determining rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    # Plot rolling statistics:
    plt.plot(timeseries, color="blue", label="Original")
    plt.plot(rolmean, color="red", label="Rolling Mean")
    plt.plot(rolstd, color="black", label="Rolling Std")
    plt.legend(loc="best")
    plt.title("Rolling Mean and Standard Deviation")
    plt.show()
    print("Results of Dickey-Fuller Test")

    adft = adfuller(timeseries, autolag="AIC")
    output = pd.Series(
        adft[0:4],
        index=[
            "Test Statistics",
            "p-value",
            "No. of lags used",
            "Number of observations used",
        ],
    )
    for key, values in adft[4].items():
        output["critical value (%s)" % key] = values
    print(output)


# Run the stationarity test
test_stationarity(data_close)

# Decompose the series to separate the trend and seasonality
result = seasonal_decompose(data_close, model="multiplicative", period=30)
fig = plt.figure()
fig = result.plot()
fig.set_size_inches(16, 9)
plt.show()

# Eliminate trend using log transformation
rcParams["figure.figsize"] = 10, 6
df_log = np.log(data_close)
moving_avg = df_log.rolling(12).mean()
std_dev = df_log.rolling(12).std()
plt.legend(loc="best")
plt.title("Moving Average")
plt.plot(std_dev, color="black", label="Standard Deviation")
plt.plot(moving_avg, color="red", label="Mean")
plt.legend()
plt.show()

# Split data into train and test sets
train_data, test_data = (
    df_log[3 : int(len(df_log) * 0.9)],
    df_log[int(len(df_log) * 0.9) :],
)
plt.figure(figsize=(10, 6))
plt.grid(True)
plt.xlabel("Dates")
plt.ylabel("Closing Prices")
plt.plot(train_data, "green", label="Train data")
plt.plot(test_data, "blue", label="Test data")
plt.legend()
plt.show()

# Use auto ARIMA to determine the best parameters
model_autoARIMA = pm.auto_arima(
    train_data,
    start_p=0,
    start_q=0,
    test="adf",  # use adftest to find optimal 'd'
    max_p=3,
    max_q=3,  # maximum p and q
    m=1,  # frequency of series
    d=None,  # let model determine 'd'
    seasonal=False,  # No Seasonality
    start_P=0,
    D=0,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True,
)
print(model_autoARIMA.summary())
model_autoARIMA.plot_diagnostics(figsize=(15, 8))
plt.show()


model = ARIMA(train_data, order=(1, 2, 2))  # ARIMA model fitting
fitted = model.fit()

# Forecast again with the adjusted model
forecast_object = fitted.get_forecast(steps=len(test_data))
fc = forecast_object.predicted_mean
conf = forecast_object.conf_int()

# Align forecast indices with test_data
fc.index = test_data.index
conf.index = test_data.index

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(train_data, label="Training Data")
plt.plot(test_data, color="blue", label="Actual Stock Price")
plt.plot(fc, color="orange", label="Predicted Stock Price")
plt.fill_between(fc.index, conf.iloc[:, 0], conf.iloc[:, 1], color="k", alpha=0.1)
plt.title("Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend(loc="upper left")
plt.show()
# Calculate Mean Squared Error (MSE)
# mse = mean_absolute_error(test_data, fc)
mse = mean_squared_error(test_data, fc)
print(f"Mean Squared Error: {mse}")

# Plot residuals
residuals = test_data - fc

plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title("Residuals of the Forecast")
plt.xlabel("Date")
plt.ylabel("Residuals")
plt.show()

# Plot residuals distribution
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=20)
plt.title("Distribution of Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.show()

# Diagnostic plots of the residuals
plt.figure(figsize=(15, 8))
plt.subplot(221)
plt.plot(residuals)
plt.title("Residuals Over Time")
plt.subplot(222)
plt.hist(residuals, bins=20)
plt.title("Residual Distribution")
plt.subplot(223)
plt.scatter(fc, residuals)
plt.title("Predicted vs Residuals")
plt.subplot(224)
plt.plot(fitted.resid)
plt.title("ARIMA Residuals")
plt.tight_layout()
plt.show()
