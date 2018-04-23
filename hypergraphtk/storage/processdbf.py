import csv
from numba import jit
from dbfread import DBF


class processdbf:

    def __init__(self, filename):
        """
        :param filename: requires full path name and file extension
        """
        self.filename = filename
        self.output = []
        self.headers = []

    @jit
    def openfile(self):
        for record in DBF(self.filename):
            row = []
            # print(record)
            for i in record.items():
                if i[0] in self.headers:
                    pass
                else:
                    self.headers.append(i[0])
                row.append(i[1])
            self.output.append(row)
        self.output.insert(0, self.headers)
        return

    # method creates
    @jit
    def add_column(self, column_name, column_data):
        self.headers.append(column_name)
        cnt = 0
        for i in self.output[1:]:
            i.append(column_data[cnt])
            cnt =  cnt + 1
        return

    @jit
    def get_column(self, column_name):
        dex = self.headers.index(column_name)
        col = ['x']
        for i in self.output[1:]:
            col.append(i[dex])
        return column_name, col[1:]

    @jit
    def update_column(self, column_name, new_data):
        data = self.get_column(column_name)
        cnt = 0
        for i in data[1]:
            self.output[1:][cnt].append(i)
            cnt = cnt + 1
        return

    @jit
    def get_columns(self, column_names):
        matrix = ['x']
        headers = ['x']
        for i in column_names:
            x = self.get_column(i)
            headers.append(x[0])
            matrix.append(x[1])
        matrix[0] = headers[1:]
        return matrix




