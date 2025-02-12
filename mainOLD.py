#from kaggle import *
import json
import csv
import numpy as np # type: ignore

import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor
#NaiveBayes
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    quant_total_mensagens = 500000 #500 mil* - 1 milhao tava demorano mt

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
                        print(f"\t[FOUND!] word: {tok:<20} toxicity rating: {termo_profanity['severity_rating']:<3} index: {_index} \nuser: {msg['username']:<20} message: {msg['content']}")
                    else:
                        print(f"  [ALSO!] word: {tok:<20} toxicity rating: {termo_profanity['severity_rating']} index: {_index}\nuser: {msg['username']:<20} message: {msg['content']}")
                    
                    msg['all_toxic_occurrences'].append((termo_profanity['text'], termo_profanity['severity_rating']))

                    print("")
        #########################

        maiorFatorToxicidade = 0
        for tuple in msg['all_toxic_occurrences']:
            if maiorFatorToxicidade < tuple[1]: #[1] = severity_rating
                maiorTermoToxico = tuple[0]     #[0] = text
                maiorFatorToxicidade = tuple[1]
        
        if maiorFatorToxicidade > 0:
            msg['toxic_tuple'] = (maiorTermoToxico, maiorFatorToxicidade)


    with open("out_mensagens_toxicas-original.txt", "w+") as arq:
        for msg in mensagens_json[0:quant_treinamento]:
            if msg['toxic_tuple'] != None:
                arq.write(f"message by {msg['username']:<20}\thas {msg['toxic_tuple'][1]} toxicity, because he said:\t\t {msg['toxic_tuple'][0]}\n")











    # print('\n\naperte enter pra continuar')
    # input()
    # os.system('clear')


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
    toxicities = [mensagem['toxic_tuple'][1] if 'toxic_tuple' in mensagem and mensagem['toxic_tuple'] is not None else 0 for mensagem in mensagens_json] #chat gebitoca

    # Convert text data to numerical data using TF-IDF vectorization
    X_train, X_test, y_train, y_test = train_test_split(
        contents,
        toxicities,
        train_size = quant_treinamento, #70%
        test_size = quant_teste, #30%
        random_state = 42
    )

    # Convert text data to numerical data using TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features = 10000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)







##################################################################################################
# todos os modelos de mantovs
##################################################################################################

    def evaluate_model(model, nome_modelo, X_train_dense = None, X_test_dense = None):
        if X_train_dense is not None and X_test_dense is not None:
            X_train_data = X_train_dense
            X_test_data = X_test_dense
        else:
            X_train_data = X_train_tfidf
            X_test_data = X_test_tfidf
        
        #treina o modelo
        model.fit(X_train_data, y_train)

        #faz a previsao
        y_pred = model.predict(X_test_data)

        #avaliar o modelo
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"\n{nome_modelo} - Mean Squared Error: {mse}")
        print(f"{nome_modelo} - R^2 Score: {r2}")

        #printa todos os valores toxicos acima de 0.3 em cada arquivo
        with open(f"out_modelos/out_mensagens_toxicas-{nome_modelo}.txt", "w+") as arq:
            for content, true_toxicity, predicted_toxicity in zip(X_test, y_test, y_pred):
                if predicted_toxicity > 0.3:
                    arq.write(f"Found message: {content}\n└--> True Toxicity: {true_toxicity:<4} Predicted Toxicity: {predicted_toxicity:.4f}\n")


    #todos os modelos
    models = [
        (Ridge(), "Ridge"),
        (SVR(), "SVR"),
        (RandomForestRegressor(), "RandomForest"),
        (DecisionTreeRegressor(), "DecisionTree"),
        ####(GaussianProcessRegressor(), "GaussianProcess"),
        (KNeighborsRegressor(), "KNN")
    ]

    for model, name in models:
        # if name == "GaussianProcess":
        #     X_train_dense = X_train_tfidf.toarray()
        #     X_test_dense = X_test_tfidf.toarray()
        #     evaluate_model(model, name, X_train_dense, X_test_dense)
        # else:
            evaluate_model(model, name)

    # Ridge - Mean Squared Error:           0.006484928537925116
    # Ridge - R^2 Score:                    0.1173933567835157

    # SVR - Mean Squared Error:             0.00952025890704332
    # SVR - R^2 Score:                     -0.2957187897070963

    # RandomForest - Mean Squared Error:    0.006890170755238344
    # RandomForest - R^2 Score:             0.06223939926174615

    # DecisionTree - Mean Squared Error:    0.008575732061624661
    # DecisionTree - R^2 Score:            -0.16716753989956223

    # KNN - Mean Squared Error:             0.007154400000000001
    # KNN - R^2 Score:                      0.02627747841792305
