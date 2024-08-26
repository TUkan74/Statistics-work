# Tesla stock price prediction using ARIMA and MSE

In this project, we predict the stock price of Tesla (TSLA) using the ARIMA (AutoRegressive Integrated Moving Average) model. We evaluate the performance of the model using the Mean Squared Error (MSE) and analyze the residuals.

## 1. Loading preprocessed data

We first need to load data about the close prices of the chosen stock

```python

data = pd.read_csv("tsla.us.txt", index_col="Date", parse_dates=True)
data_close = data["Close"]
```

## 2. Data stationarity testing

Before applying the ARIMA model, we need to ensure that the data is stationary. We will use the Dickey-Fuller test to check for stationarity while plotting the moving average and standard deviation.

```python
def test_stationarity(timeseries):
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
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
```    
![Rolling Mean and Standart Deviation](Images/Figure_1.png)
```    
Results of Dickey-Fuller Test
Test Statistics                  -0.842866
p-value                           0.806205
No. of lags used                  1.000000
Number of observations used    1856.000000
critical value (1%)              -3.433878
critical value (5%)              -2.863099
critical value (10%)             -2.567600
```
# Spustenie testu stacionarity
```python
test_stationarity(data_close)
```

## 3. Decomposition of the time series

We decompose the time series into trend, seasonality and residuals using decomposition. This will give us a better overview of the individual components of the time series.

```python
result = seasonal_decompose(data_close, model="multiplicative", period=30)
fig = plt.figure()
fig = result.plot()
fig.set_size_inches(16, 9)
plt.show()
```
![TSR](Images/Figure_2.png)

## 4. Removal of trend and application of logarithmic transformation

To remove the trend from the data, we apply a logarithmic transformation and then plot the moving average and standard deviation.

```python
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
```

![Standart Deviation](Images/Moving_avarage.png)


## 5. Division of data into training and test sets
We will divide the data into training and testing sets, using 90% of the data for training and 10% for testing.

```python
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
```

![Train and Test Data](Images/Train_Test_data.png)

## 6. Automatic selection of parameters for ARIMA using Auto ARIMA
We will use the auto_arima function to automatically determine the best parameters for the ARIMA model based on the training data.

```python
model_autoARIMA = pm.auto_arima(
    train_data,
    start_p=0,
    start_q=0,
    test="adf",  
    max_p=3,
    max_q=3,  
    m=1,  
    d=None,  
    seasonal=False,  
    start_P=0,
    D=0,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True,
)
print(model_autoARIMA.summary())
```
```
SARIMAX Results
==============================================================================
Dep. Variable:                      y   No. Observations:                 1669
Model:               SARIMAX(0, 1, 0)   Log Likelihood                3334.226
Date:                Wed, 21 Aug 2024   AIC                          -6664.451
Time:                        12:42:19   BIC                          -6653.612
Sample:                             0   HQIC                         -6660.435
                               - 1669
Covariance Type:                  opg
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
intercept      0.0015      0.001      1.869      0.062   -7.29e-05       0.003
sigma2         0.0011   1.85e-05     58.233      0.000       0.001       0.001
===================================================================================
Ljung-Box (L1) (Q):                   0.34   Jarque-Bera (JB):              2632.60
Prob(Q):                              0.56   Prob(JB):                         0.00
Heteroskedasticity (H):               0.43   Skew:                             0.13
Prob(H) (two-sided):                  0.00   Kurtosis:                         9.15
===================================================================================
```
```python
model_autoARIMA.plot_diagnostics(figsize=(15, 8))
plt.show()
```

![Diagnostics](Images/Diagnostics.png)

## 7. Training the ARIMA model with modified parameters
Based on the results from auto_arima, we adjust the parameters of the model and train it again. We then predict stock prices based on test data.

```python
model = ARIMA(train_data, order=(1, 2, 2))  
fitted = model.fit()

forecast_object = fitted.get_forecast(steps=len(test_data))
fc = forecast_object.predicted_mean
conf = forecast_object.conf_int()

fc.index = test_data.index
conf.index = test_data.index


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
```
![Prediction](Images/Prediction.png)


## 8. Model evaluation using MSE and analysis of residuals
To evaluate the model, we calculate the mean squared error (MSE) and analyze the residuals to identify potential model deficiencies.

```python
mse = mean_squared_error(test_data, fc)
print(f"Mean Squared Error: {mse}")
```
```
Mean Squared Error: 0.011125971605201944
```

```python
residuals = test_data - fc

plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title("Residuals of the Forecast")
plt.xlabel("Date")
plt.ylabel("Residuals")
plt.show()
```
![Residuals](Images/Residuals.png)
```python
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=20)
plt.title("Distribution of Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.show()
```
![Frequency](Images/distribution_of_residuals.png)
```python

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
```
![Residual Diagnostics](Images/More_Residuals.png)

# Conclusion
We evaluated the model using MSE and analyzed the residuals to identify potential problems in prediction. Although the model has clear flaws and I would not recommend it to anyone who wants to put money into the stock market, it predicted the price with good accuracy and was not far off at all. He is a bit optimistic, but in reality in 2024, he succeeded (the data is only up to 2017).

