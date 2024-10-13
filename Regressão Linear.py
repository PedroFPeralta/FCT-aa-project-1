import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

def plot_y_yhat(y_test, y_pred, plot_title="baseline-model"):
    labels = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']
    MAX = 500
    if len(y_test) > MAX:
        idx = np.random.choice(len(y_test), MAX, replace=False)
    else:
        idx = np.arange(len(y_test))

    plt.figure(figsize=(10, 10))
    for i in range(6):
        x0 = np.min(y_test[idx, i])
        x1 = np.max(y_test[idx, i])
        plt.subplot(3, 2, i + 1)
        plt.scatter(y_test[idx, i], y_pred[idx, i])
        plt.xlabel('True ' + labels[i])
        plt.ylabel('Predicted ' + labels[i])
        plt.plot([x0, x1], [x0, x1], color='red')
        plt.axis('square')

    plt.savefig('baseline.pdf')
    plt.show()



def custom_train_test_split(data, test_size=0.1, val_size=0.1, random_state=42):
    X = data[['t']]
    y = data[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']]

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    val_ratio_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_ratio_adjusted,
                                                      random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_linear_model(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    pipeline.fit(X_train, y_train)

    return pipeline

def train_polynomial_regression(X_train, y_train, degree=1):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Escalonamento
        ('poly', PolynomialFeatures(degree=degree)),  # Transformação polinomial
        ('regressor', LinearRegression())  # Regressão Linear
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def create_submission_file(pred, filename="baseline-model.csv"):
    baseline_model = pd.DataFrame(pred, columns=['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3'])

    baseline_model.insert(0, 'Id', baseline_model.index)

    baseline_model.to_csv("./baseline-model.csv", index=False)
    print(f"File Created: {filename}")

if __name__ == "__main__":
    # Split DataSet
    data = pd.read_csv('./mlNOVA/Y_train_output.csv')

    X_train, X_val, X_test, y_train, y_val, y_test = custom_train_test_split(data)

    #Train DataSet
    model = train_polynomial_regression(X_train, y_train)
    # model = train_linear_model(X_train, y_train)
    y_pred_val = model.predict(X_val)

    plot_y_yhat(y_val.values, y_pred_val)

    y_pred_test = model.predict(X_test)


    # Real Execution
    data_test = pd.read_csv('./mlNOVA/X_test.csv')
    x_data = data_test[['t']]
    y_data = data_test[['x0_1', 'y0_1', 'x0_2', 'y0_2', 'x0_3', 'y0_3']]

    data_pred = model.predict(x_data)

    # Generate Submission file
    create_submission_file(data_pred)

    mse = mean_squared_error(y_val, y_pred_val)
    rmse = root_mean_squared_error(y_val, y_pred_val)
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
