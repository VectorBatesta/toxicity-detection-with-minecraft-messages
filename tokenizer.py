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
    # nomeArquivo: str
    # separadores: list

    # quantHeader: int
    # header: list
    
    # toxicWords_amount: int
    # toxicWordLIST: list




    def __init__(self, nomeArquivo, separadores, quantHeader, toxicWords_amount):
        super().__init__(nomeArquivo, separadores)

        self.quantHeader = quantHeader
        
        self.header = []
        self.getsetHeader(quantHeader)

        self.toxicWordLIST = [] #vetor de vetores
        self.toxicWords_Amount = toxicWords_amount #quant linhas do profanity dataset
        self.grabToxic_Words()
        
        

    def getsetHeader(self, quant):
        for _ in range(quant):
            tok = self.getToken()
            self.header.append(tok)




    def grabToxic_Words(self):
        for i in range(self.toxicWords_Amount):
            node = []
            for _ in range(self.quantHeader):
                token = self.getToken()
                node.append(token)
                
            #print(node)
            self.toxicWordLIST.append(node)
        
    






class playerMessage(tipoArquivo):
    def __init__(self, nomeArquivo, separadores):
        super().__init__(nomeArquivo, separadores)
        

    