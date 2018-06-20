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
        for record in DBF(self.filename, encoding='utf-8'):
            row = []
            for i in record.items():
                if i[0] in self.headers:
                    pass
                else:
                    self.headers.append(i[0])
                row.append(i[1])
            self.output.append(row)
        self.output.insert(0, self.headers)
        return


    @jit
    def add_column(self, column_name, column_data):
        """
        :param column_name:
        :param column_data:
        :return:
        """
        self.headers.append(column_name)
        cnt = 0
        for i in self.output[1:]:
            i.append(column_data[cnt])
            cnt =  cnt + 1
        return


    @jit
    def get_column(self, column_name):
        """
        :param column_name:
        :return:
        """
        dex = self.headers.index(column_name)
        col = ['x']
        for i in self.output[1:]:
            col.append(i[dex])
        return column_name, col[1:]


    @jit
    def update_column(self, column_name, new_data):
        """
        :param column_name:
        :param new_data:

        :return:
        """
        data = self.get_column(column_name)[1]
        cnt = 0
        for i in range(len(data)):
            self.output[1:][cnt][i] = new_data[i]
            cnt = cnt + 1
        return


    @jit
    def get_columns(self, column_names):
        """
        :param column_names:
        :return:
        """
        matrix = ['x']
        headers = ['x']
        for i in column_names:
            x = self.get_column(i)
            headers.append(x[0])
            matrix.append(x[1])
        matrix[0] = headers[1:]
        return matrix


    @jit
    def get_row(self, column_name, id):
        """
        :param column_name:
        :param id:
        :return:
        """
        dex = self.headers.index(column_name)
        row = ['x']
        for i in self.output[1:]:
            if i[dex] == id:
                row.append(i)
                break
            else:
                pass
        return row[1:]


    @jit
    def add_row(self, data):
        """
        :param data:
        :return:
        """
        self.output[1:].append(data)
        return


    @jit
    def open_csv(self):
        file = open(self.filename, 'r', encoding='utf-8', errors='ignore')
        rfile = csv.reader(file)
        for i in rfile:
            self.output.append(i)
        self.headers = self.output[0]
        return


    @jit
    def save_csv(self, filename):
        """
        :param filename:
        :return:
        """
        file = open(filename, 'w', newline='')
        outfile = csv.writer(file)
        for i in self.output:
            outfile.writerow(i)
        return



