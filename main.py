#from kaggle import *
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

    for termo_profanity in profanity_list:
        termo_profanity['severity_rating'] = float(termo_profanity['severity_rating'])


    # print(list(profanity_dicts))
    # for termo in profanity_dicts:
    #     print(termo['text'], '@', termo['severity_rating'])









    quantmensagens = 1000000





    for _index, msg in enumerate(mensagens_json[0:quantmensagens]):
        msg['tokens'] = msg['content'].lower().split(" ")
        msg['all_toxic_occurrences'] = []

        #print(msg['tokens'])

        #itera entre todo token da mensagem
        for tok in msg['tokens']:
            #itera em cada termo da lista de profanity
            for termo_profanity in profanity_list:
                
                #achou token em um dos profanitys
                if tok.lower() == termo_profanity['text'].lower():
                    msg['all_toxic_occurrences'].append((termo_profanity['text'], termo_profanity['severity_rating']))

                    print("\t[FOUND!] word:", tok, "\ttoxicity rating:", termo_profanity['severity_rating'], "\tindex:", _index,
                          "\nuser:", msg['username'],
                          "\nmessage:", msg['content'])
        #########################

        maiorFatorToxicidade = 0
        for tuple in msg['all_toxic_occurrences']:
            if maiorFatorToxicidade < tuple[1]: #[1] = severity_rating
                maiorTermoToxico = tuple[0]     #[0] = text
                maiorFatorToxicidade = tuple[1]
        
        if maiorFatorToxicidade > 0:
            msg['toxic_tuple'] = (maiorTermoToxico, maiorFatorToxicidade)


    for msg in mensagens_json[0:quantmensagens]:
        if msg['toxic_tuple'] != None:
            print('message by', msg['username'], 'has', msg['toxic_tuple'][1], 'toxicity, because he said:\n', msg['toxic_tuple'][0])

    

    


    # para cada mensagem
    #     tokeniza a mensagem vira um vetor de tokens
    #     tokens estao contidos no dict profanidade
    #         vetor de true/false com 1400 valores
    #     filtra só os True
    #     verifica o maior valor entre mild strong severe
