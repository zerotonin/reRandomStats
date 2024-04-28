import pandas as pd
from prettytable import PrettyTable

def write_pretty_table(df, file_path, write=False, show = True):
    # Create a PrettyTable object
    table = PrettyTable()

    # Add columns to the table
    table.field_names = df.columns.tolist()

    # Adding rows from DataFrame
    for index, row in df.iterrows():
        table.add_row(row)

    if write:
        with open(file_path, 'w') as file:
            # Write the table to the file
            file.write(table.get_string())

    # Print the table
    if show:
        print(table)
