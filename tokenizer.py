from copy import *

class tipoArquivo:
    def __init__(self, nomeArquivo):
        self.arq = open(nomeArquivo, "r")


class tipoToken:
    def __init__(self):
        pass


class listaTokens:
    def __init__(self, arq: tipoArquivo, ordemCanonica, splitter):
        self.lista = []
        self.ordemCanonica = ordemCanonica
        self.splitter = splitter
        

    