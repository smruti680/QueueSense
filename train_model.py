"""train_model.py - trains the QueueSense model and saves it as model.pkl

Run this file once:  python train_model.py
"""
import os
import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Step 1: Load the dataset
data = pd.read_csv(os.path.join(BASE_DIR, "dataset.csv"))
print("Rows in dataset:", len(data))

# Step 2: Prepare the input (X) and the answer we want to predict (y)
# Total work for each counter = people x service time / counters
data["workload"] = data["people_waiting"] * data["avg_service_time"] / data["active_counters"]
X = data[["workload"]].values
y = data["waiting_time"].values

# Step 3: Split the data -> 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training rows:", len(X_train), "| Testing rows:", len(X_test))

# Step 4: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Test the model on data it has never seen
predictions = model.predict(X_test)
print("Average error (MAE): %.2f minutes" % mean_absolute_error(y_test, predictions))
print("R2 score: %.3f (closer to 1 is better)" % r2_score(y_test, predictions))
print("Learned formula: waiting_time = %.2f x workload + %.2f" % (model.coef_[0], model.intercept_))

# Extra: why we use "workload" instead of the 3 raw columns
raw = data[["people_waiting", "active_counters", "avg_service_time"]].values
Xr_train, Xr_test, yr_train, yr_test = train_test_split(raw, y, test_size=0.2, random_state=42)
raw_model = LinearRegression().fit(Xr_train, yr_train)
print("For comparison, using the 3 raw columns gives R2 = %.3f" % r2_score(yr_test, raw_model.predict(Xr_test)))

# Step 6: Save the trained model
with open(os.path.join(BASE_DIR, "model.pkl"), "wb") as file:
    pickle.dump(model, file)
print("Model saved as model.pkl")
