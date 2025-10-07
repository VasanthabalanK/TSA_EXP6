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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
data = pd.read_csv("World_Population.csv")

data.columns = data.columns.str.strip()

possible_cols = [col for col in data.columns if "Pop" in col or "pop" in col]
if not possible_cols:
    raise ValueError("No column found with 'Population' in its name.")
pop_col = possible_cols[0]

for col in [pop_col, "Fert. Rate", "Med. Age", "World Share"]:
    if col in data.columns:
        data[col] = (
            data[col]
            .astype(str)
            .str.replace(",", "")
            .str.replace("%", "")
            .str.replace(" ", "")
        )
        data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.dropna(subset=[pop_col, "Fert. Rate", "Med. Age", "World Share"])

X = data[["Fert. Rate", "Med. Age", "World Share"]]
y = data[pop_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
data["Predicted Population"] = model.predict(X)

mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("\nModel Performance Metrics:")
print(f"Mean Absolute Error (MAE): {mae:,.2f}")
print(f"Root Mean Square Error (RMSE): {rmse:,.2f}")

test_results = pd.DataFrame({
    "Actual Population": y_test,
    "Predicted Population": y_pred_test
}).reset_index(drop=True)

plt.figure(figsize=(10, 6))
plt.plot(test_results["Actual Population"].values, label="Actual", marker='o', color='blue')
plt.plot(test_results["Predicted Population"].values, label="Predicted", marker='x', color='red')
plt.title("Test Data: Actual vs Predicted Population")
plt.xlabel("Sample Index")
plt.ylabel("Population")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

sorted_data = data.sort_values(by=pop_col, ascending=False).reset_index(drop=True)

plt.figure(figsize=(12, 6))
plt.plot(sorted_data[pop_col].values, label="Actual Population", marker='o', color='green')
plt.plot(sorted_data["Predicted Population"].values, label="Predicted Population", marker='x', color='orange')
plt.title("Final Prediction: Actual vs Predicted Population (All Countries)")
plt.xlabel("Country Index (sorted by actual population)")
plt.ylabel("Population")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
```

### OUTPUT:

TEST_PREDICTION

<img width="846" height="545" alt="download" src="https://github.com/user-attachments/assets/f3822f71-6b4e-47a0-989d-d4c467386edf" />


FINAL_PREDICTION

<img width="1001" height="545" alt="download" src="https://github.com/user-attachments/assets/7a39637c-706c-4997-80b0-53701cd51fa9" />

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
