from copy import *


class tipoArquivo:
    def __init__(self, nomeArquivo, separadores, quantCanonics):
        self.arq = open(nomeArquivo, "r", encoding = "utf8")
        self.separadores = separadores

        self.canonics = []
        self.getsetCanonics(quantCanonics)



    def getLetra(self, quant = 1):
        return self.arq.read(quant)
    


    def getToken(self):
        tok = ""
        letra = self.getLetra()
        
        while letra not in self.separadores:
            tok = tok + letra
            letra = self.getLetra()

        return tok



    def getsetCanonics(self, quant):
        for i in range(quant):
            tok = self.getToken()
            self.canonics.append(tok)
        
        return




class tipoToken:
    def __init__(self):
        pass


class listaTokens:
    def __init__(self, arq: tipoArquivo, ordemCanonica, splitter):
        self.lista = []
        self.ordemCanonica = ordemCanonica
        self.splitter = splitter
        

    