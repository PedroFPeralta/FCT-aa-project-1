import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


def plot_y_yhat(y_test, y_pred, plot_title="baseline-model"):
    labels = ['x0_1', 'y0_1', 'x0_2', 'y0_2', 'x0_3', 'y0_3']
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

    plt.savefig(f'{plot_title}.pdf')
    plt.show()


def custom_train_test_split(X_file, Y_file, test_size=0.1, random_state=42):
    # Removing T
    Y = Y_file[['x0_1', 'y0_1', 'x0_2', 'y0_2', 'x0_3', 'y0_3']]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_file, Y, test_size=test_size)

    return X_train, X_test, y_train, y_test


def create_submission_file(pred, filename="baseline-model.csv"):
    baseline_model = pd.DataFrame(pred, columns=['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3'])
    baseline_model.insert(0, 'Id', baseline_model.index)
    baseline_model.to_csv(filename, index=False)
    print(f"File Created: {filename}")


def validate_poly_regression(X_train, y_train, X_val, y_val, regressor=None, degrees=range(1, 15), max_features=None):
    # Default model
    if regressor is None:
        regressor = LinearRegression()

    # Using only 1% of data
    X_train_sample, y_train_sample = resample(X_train, y_train, n_samples=int(0.01 * len(X_train)))

    best_rmse = float('inf')
    best_model = None

    for degree in degrees:
        poly_pipeline = make_pipeline(StandardScaler(), PolynomialFeatures(degree=degree, include_bias=False),
                                      regressor)

        # Fit the data to the pipeline
        poly_pipeline.fit(X_train_sample, y_train_sample)
        y_val_pred = poly_pipeline.predict(X_val)

        # Get the RMSE
        rmse = root_mean_squared_error(y_val, y_val_pred)

        poly_features = poly_pipeline.named_steps['polynomialfeatures']
        print(f"Degree: {degree}, Number of Features: {poly_features.n_output_features_}, RMSE: {rmse}")

        # Verify if the model is the best
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = poly_pipeline

    return best_model, best_rmse


def validate_knn_regression(X_train, y_train, X_val, y_val, k=range(1, 15)):
    # Armazenar os resultados para cada valor de K
    mse_values = []
    best_model = None
    best_k = None
    best_mse = float('inf')  # Inicializar com um valor de erro muito alto

    # Iterar sobre os diferentes valores de K
    for k_val in k:
        # Definir o modelo KNN Regressor
        knn_regressor = KNeighborsRegressor(n_neighbors=k_val)

        # Treinar o modelo com os dados de treino
        knn_regressor.fit(X_train, y_train)

        # Fazer previsões nos dados de validação
        y_pred = knn_regressor.predict(X_val)

        # Calcular o erro quadrático médio (MSE)
        mse = mean_squared_error(y_val, y_pred)

        # Armazenar o valor do MSE para este K
        mse_values.append((k_val, mse))

        # Se este K tem o menor MSE, salvar este modelo como o melhor até agora
        if mse < best_mse:
            best_mse = mse
            best_k = k_val
            best_model = knn_regressor

    print(f"O melhor valor de K é {best_k} com MSE = {best_mse}")

    # Retornar o modelo treinado com o melhor K, os valores de MSE para cada K, e o melhor K
    return best_model, best_k, best_mse


if __name__ == "__main__":
    # Import our Dataset
    X_file = pd.read_csv('./mlNOVA/X_train_output.csv')
    Y_file = pd.read_csv('./mlNOVA/Y_train_output.csv')

    models_with_rmse = []

    # Run your function 10 times and examine the distribution of the selected polynomial degrees
    for i in range(1, 11):
        print(f"Iteration {i}")
        X_train, X_test, y_train, y_test = custom_train_test_split(X_file, Y_file)
        # model, ex_best_rmse = validate_poly_regression(X_train, y_train, X_test, y_test, regressor=Ridge(0.1), degrees=range(1, 9))
        model, best_k, ex_best_rmse = validate_knn_regression(X_train, y_train, X_test, y_test, k=range(1, 10))

        models_with_rmse.append({'model': model, 'rmse': ex_best_rmse, 'k': best_k})

    # Order to get best model, rmse and k
    best_model_info = min(models_with_rmse, key=lambda x: x['rmse'])
    best_model = best_model_info['model']
    best_rmse = best_model_info['rmse']
    best_k = best_model_info['k']

    print(f"Best RMSE: {best_rmse} | Best K {best_k}")

    #Plot best model
    aux_x_train, aux_x_test, aux_y_train, aux_y_test = custom_train_test_split(X_file, Y_file)
    plot_y_yhat(aux_y_test.values, best_model.predict(aux_x_test))

    # Using Test data provided by kagle
    data_test = pd.read_csv('./mlNOVA/X_test.csv')
    x_data_test = data_test.drop(columns=['Id'])
    data_pred = best_model.predict(x_data_test)

    # Generating Submission File
    create_submission_file(data_pred, filename=f"knn_submission.csv")