import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
data = pd.read_csv("co2.csv")

def create_ts_data(data, window_size, target_size):
    i = 1
    while i < window_size:
        data[f"co2_{i}"] = data["co2"].shift(-i)
        i += 1
    i = 0
    while i < target_size:
        data[f"target_{i}"] = data["co2"].shift(-i - window_size)
        i += 1

    # axis = 0 (drop by rows)
    data = data.dropna(axis = 0)

    return data

#print(data.info())
# Since the type of time column is object, we want to convert to date time
data["time"] = pd.to_datetime(data["time"])
#print(data.info())

# In numerical analysis, interpolation is a method of estimating values between known data points.
data["co2"] = data["co2"].interpolate()

fig, ax = plt.subplots()

# # (x axis, y axis)
# ax.plot(data["time"], data["co2"])
# ax.set_xlabel("Time")
# ax.set_ylabel("CO2")
# plt.show()

window_size = 5
train_ratio = 0.8
target_size = 3

data = create_ts_data(data, window_size, target_size)
num_samples = len(data)

targets = [f"target_{i}" for i in range(target_size)]

x = data.drop(["time"] + targets , axis=1)
y = data[targets]

x_train = x[:int(train_ratio * num_samples)]
y_train = y[:int(train_ratio * num_samples)]

x_test = x[int(train_ratio * num_samples):]
y_test = y[int(train_ratio * num_samples):]

# models = [LinearRegression() for _ in range(target_size)]
models = [LinearRegression(), RandomForestRegressor()]

# r2 = []
# mae = []
# mse = []

for i, model in enumerate(models):
    r2 = []
    mae = []
    mse = []

    if isinstance(model, LinearRegression):
        print("Linear Regression")
        model.fit(x_train, y_train[f"target_{i}"])
        y_predict = model.predict(x_test)
        r2.append(r2_score(y_test[f"target_{i}"], y_predict))
        mae.append(mean_absolute_error(y_test[f"target_{i}"], y_predict))
        mse.append(mean_squared_error(y_test[f"target_{i}"], y_predict))
        print(f"MAE: {mae}")
        print(f"MSE: {mse}")
        print(f"R2: {r2}")
        print()
    else:

        print("Random Forest")
        model.fit(x_train, y_train[f"target_{i}"])
        y_predict = model.predict(x_test)
        r2.append(r2_score(y_test[f"target_{i}"], y_predict))
        mae.append(mean_absolute_error(y_test[f"target_{i}"], y_predict))
        mse.append(mean_squared_error(y_test[f"target_{i}"], y_predict))
        print(f"MAE: {mae}")
        print(f"MSE: {mse}")
        print(f"R2: {r2}")

# print(data.corr())
#
# model = RandomForestRegressor(random_state=42)
# model.fit(x_train, y_train)
#
# y_predicted = model.predict(x_test)
#
# # Evaluate using different metrics
# print(f"MAE: {mean_absolute_error(y_test, y_predicted)}")
# print(f"MSE: {mean_squared_error(y_test, y_predicted)}")
# print(f"RMSE: {root_mean_squared_error(y_test, y_predicted)}")
# print(f"R^2: {r2_score(y_test, y_predicted)}")

# (x axis, y axis)
# ax.plot(data["time"], data["co2"])
# ax.plot(data["time"][:int(train_ratio * num_samples)], y_train, label = "train")
# ax.plot(data["time"][int(train_ratio * num_samples):], y_test, label = "test")
# ax.plot(data["time"][int(train_ratio * num_samples):], y_predicted, label = "predict")
# ax.set_xlabel("Time")
# ax.set_ylabel("CO2")
# ax.legend()
# ax.grid()
# plt.show()