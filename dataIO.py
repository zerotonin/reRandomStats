import csv
import numpy as np

class DataIO:
    """
    This class is used to handle input and output operations on data.
    """

    def __init__(self):
        """
        Initialize the class and preallocate variables.
        """
        self.raw_data = None

    def read_csv(self, file_path):
        """
        Reads in a CSV file and saves it to the `raw_data` variable.

        Parameters:
            file_path (str): The path to the CSV file.
        """
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            # the replace is needed because German Excel uses ; instead of , to seperate. Thanks Bill!
            self.raw_data = [row[0].replace(";",",") for row in reader]

    def make_square_np_matrix(self, values):
        """
        Creates a square NumPy matrix from the `values` parameter.

        Parameters:
            values (list): A list of strings, where each string represents a row in the matrix.

        Returns:
            numpy.ndarray: A square matrix with `np.nan` values for empty cells.
        """
        nrows = len(values)
        ncols = max([row.count(",") for row in values]) + 1
        values = [row.split(",") for row in values]
        matrix = np.empty((nrows, ncols))
        matrix[:] = np.nan
        for i, row in enumerate(values):
            for j, val in enumerate(row):
                if val:
                    matrix[i, j] = val
        return matrix

    def split_csv_headers(self):
        """
        Splits the headers of the CSV file and returns them as a list.

        Returns:
            list: A list of headers.
        """
        return self.raw_data[0].split(",")

    def wide_table_to_value_id_list(self, values, col_header):
        """
        Converts a wide table format to a long table format, where the first column is the values and the second column is the column headers.

        Parameters:
            values (numpy.ndarray): A NumPy matrix containing the data in wide table format.
            col_header (list): A list of headers for the wide table.

        Returns:
            tuple: A tuple containing two lists, the first is the headers, the second is the values.
        """
        value_list = []
        id_list = []

        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                if not np.isnan(values[i,j]):
                    value_list.append(values[i,j])
                    id_list.append(col_header[j])

        return id_list, value_list

    def wide_table_csv_to_long_table(self, file_path):
        """
        Converts a wide table CSV file to a long table format, where the first column is the values and the second column is the column headers.

        Parameters:
            file_path (str): The path to the CSV file in wide table format.

        tuple: A tuple containing two lists, the first is the headers, the second is the values.
        """
        self.read_csv(file_path)
        data = self.make_square_np_matrix(self.raw_data[1::])
        id_str = self.split_csv_headers()
        id_list, value_list = self.wide_table_to_value_id_list(data, id_str)
        return (id_list, value_list)





#x = DataIO()  
#print(x.wide_table_csv_to_long_table("/home/bgeurten/ownCloud/Anne_Stats/SEM_SEL_Development.csv" ))  

