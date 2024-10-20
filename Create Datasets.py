import csv
import pandas as pd


def read_csv_and_write_output(input_file_train, output_file_name1, output_file_name2):
    # What is the expected format from X and Y Files
    expected_format_test = ['t', 'x0_1', 'y0_1', 'x0_2', 'y0_2', 'x0_3', 'y0_3']
    expected_format_train = ['t', 'x0_1', 'y0_1', 'x0_2', 'y0_2', 'x0_3', 'y0_3']

    df_train = pd.read_csv(input_file_train)

    # Remove velocity columns
    df_train_cleaned = df_train.drop(columns=[col for col in df_train.columns if 'v_' in col])

    # Open and write the new output files
    with open(output_file_name1, 'w', newline='') as output_file1, open(output_file_name2, 'w',
                                                                        newline='') as output_file2:
        output_writer1 = csv.writer(output_file1)
        output_writer2 = csv.writer(output_file2)

        # Write the header
        output_writer1.writerow(expected_format_train)
        output_writer2.writerow(expected_format_test)

        current_iteration_data = []
        first_values = [0] * 7  # 7 columns

        for index, row in df_train_cleaned.iterrows():
            current_time = float(row.iloc[0])

            if current_time == 0.0:

                if current_iteration_data:

                    collision_found = False
                    for iteration_row in current_iteration_data:
                        x1, y1 = float(iteration_row.iloc[1]), float(iteration_row.iloc[2])
                        x2, y2 = float(iteration_row.iloc[3]), float(iteration_row.iloc[4])
                        x3, y3 = float(iteration_row.iloc[5]), float(iteration_row.iloc[6])

                        if (x1, y1) == (x2, y2) == (x3, y3):
                            collision_found = True
                            break

                    if not collision_found:
                        for iteration_row in current_iteration_data:
                            converted_row1 = [iteration_row.iloc[0]] + first_values
                            output_writer1.writerow(converted_row1)

                            converted_row2 = [iteration_row.iloc[0]] + [float(iteration_row.iloc[i]) for i in
                                                                        range(1, 7)]
                            output_writer2.writerow(converted_row2)

                current_iteration_data = []
                first_values = [float(row.iloc[i]) for i in range(1, 7)]

            current_iteration_data.append(row)

if __name__ == "__main__":
    read_csv_and_write_output('./mlNOVA/X_train.csv', './mlNOVA/X_train_output.csv',
                              './mlNOVA/Y_train_output.csv')

