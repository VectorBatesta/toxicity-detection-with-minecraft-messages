#from kaggle import *
from tokenizer import *
import json

"""
[
    {
        "username": "Cpl_iPatch",
        "content": "everyone come to 700 50k its my base",
        "date": "09/02/2020 2:14 AM"
    },
    {
"""

[TODO: json]

muito complicado \/

# def cleanDataset():
#     arq = open("clean.json", "r", encoding = "utf8")
#     clean = open("clean.txt", "w+", encoding = "utf8")
    
#     while True:
#         txt = arq.read(1)

#         if txt == '': #eof em python eh emptytext
#             break

#         while txt != "\"": #acha o primeiro "username"
#             txt = arq.read(1)
        
#         arq.read(13) #'"username": "' tem 13 char

#         txt = '' #reseta
#         letra = ''
#         while letra != '\"': #le ate acabar nickname
#             letra = arq.read(1)
#             txt.append(letra)
            
#         clean.write(txt) #escreve nome
#         clean.write('@') #nao tem @ no arquivo, usar como separador

#         arq.read(24) #24 chars ate texto do usuario


        




if __name__ == "__main__":
    cleanDataset()


if __name__ == "__amain__":
    profanity = profanityArquivo(
        nomeArquivo = "profanity_en.csv",
        separadores = [',', '\n'], #so separar tokens por virgula por causa do CSV
        quantHeader = 9, #quantidade de tokens no header
        toxicWords_amount = 1597 #1598 linhas em profanity tirando o header
    ) 
    print(profanity.header)

    print(profanity.toxicWordLIST[0])
    
    

    #lista_profanidade = tokenizer(fil)