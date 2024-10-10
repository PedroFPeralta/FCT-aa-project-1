import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import csv

def read_csv_and_write_output(input_file_name, output_file_name1, output_file_name2):
    # Definimos o formato esperado do CSV, mas vamos remover as colunas de velocidade
    expected_format = ['t', 'x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']

    # Usamos pandas para ler o CSV
    df = pd.read_csv(input_file_name)

    # Removemos as colunas que têm "v_" no nome (indicando velocidade)
    df_cleaned = df.drop(columns=[col for col in df.columns if 'v_' in col])

    # Abrir os dois arquivos de saída
    with open(output_file_name1, 'w', newline='') as output_file1, open(output_file_name2, 'w', newline='') as output_file2:
        output_writer1 = csv.writer(output_file1)
        output_writer2 = csv.writer(output_file2)

        # Escrevemos o cabeçalho sem as colunas de velocidade
        output_writer1.writerow(expected_format)
        output_writer2.writerow(expected_format)

        current_iteration_data = []
        first_values = [0] * 7  # Agora temos 7 colunas (t, x_1, y_1, x_2, y_2, x_3, y_3)

        for index, row in df_cleaned.iterrows():
            current_time = float(row.iloc[0])  # Corrigido usando iloc

            if current_time == 0.0:

                if current_iteration_data:

                    collision_found = False
                    for iteration_row in current_iteration_data:
                        x1, y1 = float(iteration_row.iloc[1]), float(iteration_row.iloc[2])  # Usando iloc
                        x2, y2 = float(iteration_row.iloc[3]), float(iteration_row.iloc[4])  # Usando iloc
                        x3, y3 = float(iteration_row.iloc[5]), float(iteration_row.iloc[6])  # Usando iloc

                        # if (x1, y1) == (x2, y2) == (x3, y3):
                        #     collision_found = True
                        #     break

                    if not collision_found:
                        for iteration_row in current_iteration_data:
                            converted_row1 = [iteration_row.iloc[0]] + first_values  # Usando iloc
                            output_writer1.writerow(converted_row1)

                            converted_row2 = [iteration_row.iloc[0]] + [float(iteration_row.iloc[i]) for i in range(1, 7)]  # Usando iloc
                            output_writer2.writerow(converted_row2)

                current_iteration_data = []
                first_values = [float(row.iloc[i]) for i in range(1, 7)]  # Atualizado para usar iloc

            current_iteration_data.append(row)

        if current_iteration_data:
            collision_found = False
            for iteration_row in current_iteration_data:
                x1, y1 = float(iteration_row.iloc[1]), float(iteration_row.iloc[2])  # Usando iloc
                x2, y2 = float(iteration_row.iloc[3]), float(iteration_row.iloc[4])  # Usando iloc
                x3, y3 = float(iteration_row.iloc[5]), float(iteration_row.iloc[6])  # Usando iloc

                # if (x1, y1) == (x2, y2) == (x3, y3):
                #     collision_found = True
                #     break

            if not collision_found:
                for iteration_row in current_iteration_data:
                    converted_row1 = [iteration_row.iloc[0]] + first_values  # Usando iloc
                    output_writer1.writerow(converted_row1)

                    converted_row2 = [iteration_row.iloc[0]] + [float(iteration_row.iloc[i]) for i in range(1, 7)]  # Usando iloc
                    output_writer2.writerow(converted_row2)


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


def train_linear_model(X_train_file, Y_train_file):
    # Carregar os dados
    X = pd.read_csv(X_train_file)
    y = pd.read_csv(Y_train_file)

    # Remover as colunas de velocidade de 'y' (velocidade não interessa para o estudo)
    y = y[['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']]

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Definir o pipeline de regressão
    pipeline_X = Pipeline([
        ('scaler', StandardScaler()),
        ("poly", PolynomialFeatures(degree=3)),
        ('regressor', LinearRegression())
    ])

    # Escalonador para os dados de saída (y)
    pipeline_y = StandardScaler()

    # Escalonar e treinar o modelo
    y_train_scaled = pipeline_y.fit_transform(y_train)
    pipeline_X.fit(X_train, y_train_scaled)

    # Fazer previsões no conjunto de teste
    y_pred_scaled = pipeline_X.predict(X_test)
    y_pred = pipeline_y.inverse_transform(y_pred_scaled)

    # Verificar o formato do y_pred
    print(f"y_pred shape: {y_pred.shape}")

    # Criar o DataFrame com as previsões
    baseline_model = pd.DataFrame(y_pred, columns=['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3'])

    # Adicionar o ID (índice da linha)
    baseline_model.insert(0, 'Id', baseline_model.index)

    # Salvar o DataFrame em CSV, sem incluir o índice
    baseline_model.to_csv("./baseline-model.csv", index=False)

    # Gerar o gráfico de comparação
    plot_y_yhat(y_test.values, y_pred, plot_title="baseline-model")

    # Calcular e imprimir o erro quadrático médio (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    return pipeline_X


if __name__ == "__main__":
    read_csv_and_write_output('./mlNOVA/X_train.csv', './mlNOVA/X_train_output.csv', './mlNOVA/Y_train_output.csv')
    model = train_linear_model('./mlNOVA/X_train_output.csv', './mlNOVA/Y_train_output.csv')
