import os
import json
import csv
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Set seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

if __name__ == "__main__":
    set_seeds()

    # Disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    arq_mensagens = open("clean.json", "r", encoding="utf8")

    mensagenstxt = arq_mensagens.read()
    mensagens_json = json.loads(mensagenstxt)

    arq_profanity = open("profanity_en.csv", "r", encoding="utf8")
    profanity_dicts = csv.DictReader(arq_profanity)
    profanity_list = list(profanity_dicts)

    for termo_profanity in profanity_list:
        termo_profanity['severity_rating'] = float(termo_profanity['severity_rating'])

    quant_total_mensagens = 1000  # Reduced dataset size for faster runtime
    quant_treinamento = int((quant_total_mensagens / 100) * 70)  # = 70%
    quant_teste = int((quant_total_mensagens / 100) * 30)  # = 30%

    for _index, msg in enumerate(mensagens_json[0:quant_treinamento]):
        msg['tokens'] = msg['content'].lower().split(" ")
        msg['all_toxic_occurrences'] = []
        msg['toxic_tuple'] = (None, 0)

        for tok in msg['tokens']:
            for termo_profanity in profanity_list:
                if tok.lower() == termo_profanity['text'].lower():
                    if len(msg['all_toxic_occurrences']) == 0:
                        print(f"\t[FOUND!] word: {tok:<20} toxicity rating: {termo_profanity['severity_rating']:<3} index: {_index} \nuser: {msg['username']:<20} message: {msg['content']}")
                    else:
                        print(f"  [ALSO!] word: {tok:<20} toxicity rating: {termo_profanity['severity_rating']} index: {_index}\nuser: {msg['username']:<20} message: {msg['content']}")

                    msg['all_toxic_occurrences'].append((termo_profanity['text'], termo_profanity['severity_rating']))

                    print("")

        maiorFatorToxicidade = 0
        for tuple in msg['all_toxic_occurrences']:
            if maiorFatorToxicidade < tuple[1]:  # [1] = severity_rating
                maiorTermoToxico = tuple[0]  # [0] = text
                maiorFatorToxicidade = tuple[1]

        if maiorFatorToxicidade > 0:
            msg['toxic_tuple'] = (maiorTermoToxico, maiorFatorToxicidade)

    with open("out_mensagens_toxicas-original.txt", "w+") as arq:
        for msg in mensagens_json[0:quant_treinamento]:
            if msg['toxic_tuple'] != None:
                arq.write(f"message by {msg['username']:<20}\thas {msg['toxic_tuple'][1]} toxicity, because he said:\t\t {msg['toxic_tuple'][0]}\n")

    contents = [mensagem['content'] for mensagem in mensagens_json]
    toxicities = [mensagem['toxic_tuple'][1] if 'toxic_tuple' in mensagem and mensagem['toxic_tuple'] is not None else 0 for mensagem in mensagens_json]

    new_terms_df = pd.read_csv('toxicity_en-newterms.csv')
    new_contents = new_terms_df['text'].tolist()
    new_toxicities = [1 if label == 'Toxic' else 0 for label in new_terms_df['is_toxic']]

    combined_contents = contents + new_contents
    combined_toxicities = toxicities + new_toxicities

    X_train, X_test, y_train, y_test = train_test_split(
        combined_contents,
        combined_toxicities,
        train_size=quant_treinamento,  # 70%
        test_size=quant_teste,  # 30%
        random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    X_train_dense = X_train_tfidf.toarray()
    X_test_dense = X_test_tfidf.toarray()

    X_train_tensor = tf.convert_to_tensor(X_train_dense, dtype=tf.float32)
    X_test_tensor = tf.convert_to_tensor(X_test_dense, dtype=tf.float32)
    y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float32)

    model = Sequential([
        Input(shape=(X_train_tensor.shape[1],)),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1)  # Output layer for regression
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mean_squared_error'])

    history = model.fit(X_train_tensor, y_train_tensor, epochs=5, batch_size=32, validation_split=0.2)  # Reduced epochs for faster runtime

    loss, mse = model.evaluate(X_test_tensor, y_test_tensor)
    print(f"Mean Squared Error: {mse}")