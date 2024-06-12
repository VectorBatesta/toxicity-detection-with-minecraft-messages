#from kaggle import *
from tokenizer import *
import json
import csv

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


        




"""
[
    {
        "username": "Cpl_iPatch",
        "content": "everyone come to 700 50k its my base",
        "date": "09/02/2020 2:14 AM"
    },
    {
        "username": "EpikJasper",
        "content": "cap",
        "date": "09/02/2020 2:14 AM"
    },
    {
"""


    



if __name__ == "__main__":
    profanity = profanityArquivo(
        nomeArquivo = "profanity_en.csv",
        separadores = [',', '\n'], #so separar tokens por virgula por causa do CSV
        quantHeader = 9, #quantidade de tokens no header
        toxicWords_amount = 1597 #1598 linhas em profanity tirando o header
    ) 
    print(profanity.header)

    #print(profanity.toxicWordLIST[0])








    arq = open("clean.json", "r", encoding = "utf8")

    texto = arq.read()
    texto_json = json.loads(texto)

    # for i in len(texto_json):
    #     print( texto_json[i]['username'] + ',' )
    #     print( texto_json[i]['content'] )
    
    




    # achado internet
    with open("profanity_en.csv", "r", encoding = "utf8") as file:
        csv_reader = csv.DictReader(file)

        print(dict(csv_reader))


        # for termo in csv_reader:
        #     print(termo['text'], termo['severity_rating'])

