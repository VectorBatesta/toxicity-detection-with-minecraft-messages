#from kaggle import *
from tokenizer import *
import json
import csv
import numpy as np # type: ignore


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
    
    arq_mensagens = open("clean.json", "r", encoding = "utf8")

    mensagenstxt = arq_mensagens.read()
    mensagens_json = json.loads(mensagenstxt)

    # for msg in mensagens_json:
    #     print(msg['username'], '@', msg['content'])
    
    


    # achado internet
    arq_profanity = open("profanity_en.csv", "r", encoding = "utf8")
    profanity_dicts = csv.DictReader(arq_profanity)
    profanity_list = list(profanity_dicts)

    for termo in profanity_list:
        termo['severity_rating'] = float(termo['severity_rating'])


    # print(list(profanity_dicts))
    # for termo in profanity_dicts:
    #     print(termo['text'], '@', termo['severity_rating'])


    for msg in mensagens_json[0:100000]:
        msg['tokens'] = msg['content'].split(" ")
        msg['toxicity'] = 0

        #print(msg['tokens'])

        for tok in msg['tokens']:
            for termo in profanity_list:
                if tok == termo['canonical_form_1']:
                    msg['toxicity'] += termo['severity_rating']
                    print(msg['toxicity'])


    for msg in mensagens_json[0:100000]:
        if msg['toxicity'] > 0:
            print('message by', msg['username'], 'has', msg['toxicity'], 'toxicity.')

    

    


    # para cada mensagem
    #     tokeniza a mensagem vira um vetor de tokens
    #     tokens estao contidos no dict profanidade
    #         vetor de true/false com 1400 valores
    #     filtra só os True
    #     verifica o maior valor entre mild strong severe
