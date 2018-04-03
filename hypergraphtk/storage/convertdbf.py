from dbfread import DBF

class dbfile:

    def __init__(self, filename):
        self.filename =  filename
        self.output = []

    def openfile(self):
        for record in DBF(self.filename):
            print(record)