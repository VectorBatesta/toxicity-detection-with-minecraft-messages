from copy import *


class tipoArquivo:
    def __init__(self, nomeArquivo, separadores):
        self.arq = open(nomeArquivo, "r", encoding = "utf8")
        self.separadores = separadores



    def getLetra(self, quant = 1):
        return self.arq.read(quant)
    


    def getToken(self):
        tok = ""
        letra = self.getLetra()
        
        while letra not in self.separadores:
            tok = tok + letra
            letra = self.getLetra()

        return tok



    


####################################3




class profanityArquivo(tipoArquivo):
    def __init__(self, nomeArquivo, separadores, quantHeader):
        super().__init__(nomeArquivo, separadores)
        
        self.header = []
        self.getsetHeader(quantHeader)
        
        

    def getsetHeader(self, quant):
        for _ in range(quant):
            tok = self.getToken()
            self.header.append(tok)
        
        return
        

    