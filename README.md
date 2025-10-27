# Ex.No: 6               HOLT WINTERS METHOD
### Date: 06.10.2025

### AIM:
To create and implement Holt Winter's Method Model using python for World Population dataset.
### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Load dataset
data = pd.read_csv('/content/jee_mains_2013_to_2025_top_30_ranks.csv')

# Focus on 'Year' and 'Total_Marks'
jee_data = data[['Year', 'Total_Marks']]

# Step 1: Aggregate by Year (average Total Marks)
jee_yearly = jee_data.groupby('Year')['Total_Marks'].mean().reset_index()

# Convert 'Year' to datetime for time-series
jee_yearly['Date'] = pd.to_datetime(jee_yearly['Year'], format='%Y')
jee_yearly.set_index('Date', inplace=True)

# Step 2: Resample yearly (though data already yearly, this keeps structure)
jee_yearly = jee_yearly.resample('YS').mean()

# Preview
print(jee_yearly.head())

# Step 3: Plot original data
jee_yearly.plot(figsize=(10, 6))
plt.title('Average JEE Total Marks per Year (Top 30)')
plt.ylabel('Average Total Marks')
plt.xlabel('Year')
plt.grid(True)
plt.show()

# Step 4: Scale the Total Marks column
scaler = MinMaxScaler()
scaled_marks = scaler.fit_transform(jee_yearly['Total_Marks'].values.reshape(-1, 1)).flatten()

# Create time series
scaled_data = pd.Series(scaled_marks, index=jee_yearly.index)

# Step 5: Plot scaled data
plt.figure(figsize=(10, 6))
scaled_data.plot()
plt.title('Scaled JEE Total Marks (Top 30)')
plt.ylabel('Scaled Value')
plt.xlabel('Year')
plt.grid(True)
plt.show()

# Step 6: Decompose data to view trend/seasonality
if len(scaled_data) >= 6:  # Decomposition needs enough data points
    decomposition = seasonal_decompose(scaled_data, model="additive", period=3)
    decomposition.plot()
    plt.show()

# Step 7: Train-Test Split (80-20)
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

plt.figure(figsize=(10, 6))
train_data.plot(label='Train Data')
test_data.plot(label='Test Data')
plt.legend()
plt.title('Train-Test Split')
plt.grid(True)
plt.show()

# Step 8: Holt-Winters Model
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=3).fit()

# Forecast for test period
test_predictions = model_add.forecast(steps=len(test_data))

# Plot predictions
plt.figure(figsize=(10, 6))
train_data.plot(label='Train Data')
test_data.plot(label='Test Data')
test_predictions.plot(label='Predictions', linestyle='--')
plt.legend()
plt.title('Holt-Winters Model Predictions (JEE Marks)')
plt.grid(True)
plt.show()

# Step 9: Evaluate Model
rmse = np.sqrt(mean_squared_error(test_data, test_predictions))
mae = mean_absolute_error(test_data, test_predictions)
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')

# Step 10: Train final model on full dataset
final_model = ExponentialSmoothing(scaled_data, trend='add', seasonal='add', seasonal_periods=3).fit()

# Forecast next 3 years
final_predictions = final_model.forecast(steps=3)

# Inverse transform back to original marks scale
forecast_original = scaler.inverse_transform(final_predictions.values.reshape(-1, 1)).flatten()

# Future years for plotting
future_years = pd.date_range(start=jee_yearly.index[-1] + pd.DateOffset(years=1), periods=3, freq='YS')

# Step 11: Plot results
plt.figure(figsize=(10, 6))
jee_yearly['Total_Marks'].plot(label='Actual Data')
plt.plot(future_years, forecast_original, label='Future Predictions', linestyle='--', marker='o')
plt.legend()
plt.title('Forecasted Average JEE Total Marks (Next 3 Years)')
plt.ylabel('Average Total Marks')
plt.xlabel('Year')
plt.grid(True)
plt.show()

# Step 12: Print predicted values
forecast_df = pd.DataFrame({
    'Year': future_years.year,
    'Predicted_Avg_Total_Marks': forecast_original
})
print("\nPredicted Average JEE Total Marks for Next 3 Years:")
print(forecast_df)

```

### OUTPUT:
<img width="659" height="414" alt="Screenshot 2025-10-27 224606" src="https://github.com/user-attachments/assets/5d9afde4-b3a7-4316-b988-41de1d56b71a" />
<img width="846" height="547" alt="image" src="https://github.com/user-attachments/assets/86689160-b6aa-42f8-8fcf-9a5f69fbf9ea" />
<img width="804" height="615" alt="Screenshot 2025-10-27 224753" src="https://github.com/user-attachments/assets/8ba04930-0f21-4f9f-8f8d-494a5b4308d3" />
<img width="826" height="547" alt="image" src="https://github.com/user-attachments/assets/76f94152-9424-48bc-8da7-412ced16646d" />
<img width="826" height="547" alt="image" src="https://github.com/user-attachments/assets/f85aec90-9b93-4cb3-a709-4796bf5658b4" />
<img width="1105" height="739" alt="Screenshot 2025-10-27 224809" src="https://github.com/user-attachments/assets/11a91d8c-618b-47c2-bf9f-f2f0edb9a771" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
