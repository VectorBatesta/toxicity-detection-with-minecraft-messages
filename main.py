#from kaggle import *
import json
import csv
import numpy as np # type: ignore

import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


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





    # quant_total_mensagens = len(mensagens_json)
    quant_total_mensagens = 1000000 #1 milhao

    quant_treinamento = int((quant_total_mensagens/100) * 70)  # = 70%
    quant_teste = int((quant_total_mensagens/100) * 30)        # = 30%




    # para cada mensagem
    #     tokeniza a mensagem vira um vetor de tokens
    #     tokens estao contidos no dict profanidade
    #         vetor de true/false com 1400 valores <- nao feito
    #     filtra só os True
    #     verifica o maior valor entre mild strong severe
    for _index, msg in enumerate(mensagens_json[0:quant_treinamento]):
        msg['tokens'] = msg['content'].lower().split(" ")
        msg['all_toxic_occurrences'] = []
        msg['toxic_tuple'] = (None, 0)    #tupla = (termo toxico, toxicidade)

        #print(msg['tokens'])

        #itera entre todo token da mensagem
        for tok in msg['tokens']:
            #itera em cada termo da lista de profanity
            for termo_profanity in profanity_list:
                
                #achou token em um dos profanitys
                if tok.lower() == termo_profanity['text'].lower():
                    if len(msg['all_toxic_occurrences']) == 0:
                        print("\t[FOUND!] word:", tok, "\t\ttoxicity rating:", termo_profanity['severity_rating'], "\tindex:", _index,
                              "\nuser:", msg['username'], "message:", msg['content'])
                    else:
                        print("  [ALSO!] word:", tok, "\t\ttoxicity rating:", termo_profanity['severity_rating'], "\tindex:", _index,
                              "\nuser:", msg['username'], "message:", msg['content'])
                    
                    msg['all_toxic_occurrences'].append((termo_profanity['text'], termo_profanity['severity_rating']))
        #########################

        maiorFatorToxicidade = 0
        for tuple in msg['all_toxic_occurrences']:
            if maiorFatorToxicidade < tuple[1]: #[1] = severity_rating
                maiorTermoToxico = tuple[0]     #[0] = text
                maiorFatorToxicidade = tuple[1]
        
        if maiorFatorToxicidade > 0:
            msg['toxic_tuple'] = (maiorTermoToxico, maiorFatorToxicidade)


    for msg in mensagens_json[0:quant_treinamento]:
        if msg['toxic_tuple'] != None:
            print('message by', msg['username'], '\thas', msg['toxic_tuple'][1], 'toxicity, because he said:\t\t', msg['toxic_tuple'][0])











    print('\n\naperte enter pra continuar')
    input()
    os.system('clear')


    # for _index, msg in enumerate(mensagens_json[0:quant_treinamento]):
    #     #printar index e quant toxicidade
    #     print(_index, ": ", msg['toxic_tuple'][1])
    #
    # print('\n\naperte enter pra continuar')
    # input()
    # os.system('clear')



##################################################################################################
# scikit goes here
##################################################################################################

    contents = [mensagem['content'] for mensagem in mensagens_json]
    toxicities = [mensagem['toxic_tuple'][1] if 'toxic_tuple' in mensagem and mensagem['toxic_tuple'] is not None else 0 for mensagem in mensagens_json]

    # Convert text data to numerical data using TF-IDF vectorization
    X_train, X_test, y_train, y_test = train_test_split(
        contents,
        toxicities,
        train_size = quant_treinamento,
        test_size = quant_teste,
        random_state = 42
    )

    # Convert text data to numerical data using TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=10000)  # Adjust max_features as needed
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train a regression model
    model = Ridge()  # You can try other regression models as well
    model.fit(X_train_tfidf, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test_tfidf)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Print all messages and their predicted toxicity values
    for content, true_toxicity, predicted_toxicity in zip(X_test, y_test, y_pred):
        if predicted_toxicity > 0.1:
            print(f"Found message: {content}\n └-->True Toxicity: {true_toxicity}\t\tPredicted Toxicity: {predicted_toxicity:.4f}")

    # Example: Predicting the toxicity of new messages
    new_messages = ["Example message 1", "Example message 2"]
    new_messages_tfidf = vectorizer.transform(new_messages)
    predicted_toxicities = model.predict(new_messages_tfidf)
    print(predicted_toxicities)