import os
import json
import csv
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

import matplotlib
matplotlib.use('Agg')

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5/plugins'


#########################
##### Configuration #####
#########################
# SEED = Seed for reproducibility
# QUANT_TOTAL_MENSAGENS = Increasing this will make training take longer but can improve model performance
# TRAIN_SIZE_RATIO = Higher ratio means more data for training, potentially better results but longer training time
# TEST_SIZE_RATIO = Lower ratio means less data for testing, potentially less reliable evaluation
# VOCAB_SIZE = Quantity of words to consider in the vocabulary
# MAX_LENGTH = Quantity of words to consider in each message
# EPOCHS = More epochs can improve model performance but increases training time
# BATCH_SIZE = Larger batch size can speed up training but may require more memory and can affect model convergence
# LEARNING_RATE = Higher learning rate can speed up training but may cause the model to converge too quickly to a suboptimal solution
# PATIENCE = Higher patience allows the model to train longer before stopping, potentially improving performance but increasing training time
SEED = 42
QUANT_TOTAL_MENSAGENS = 3000

TRAIN_SIZE_RATIO = 0.7
TEST_SIZE_RATIO = 1 - TRAIN_SIZE_RATIO

VOCAB_SIZE = 10000
MAX_LENGTH = 50

EPOCHS = 5
BATCH_SIZE = 64

LEARNING_RATE = 0.0005
PATIENCE = 3
#########################
#########################
#########################



def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

if __name__ == "__main__":
    #seta as seeds
    set_seeds()

    #diz pra OS nao usar CUDA (tirar o spam de mensagens)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    #carrega as mensagens do dataset
    with open("clean.json", "r", encoding="utf8") as f:
        mensagens_json = json.load(f)

    #carrega a lista de termos toxicos
    # with open("profanity_en.csv", "r", encoding="utf8") as f:
    #     profanity_list = list(csv.DictReader(f))
    # for termo in profanity_list:
    #     termo['severity_rating'] = float(termo['severity_rating'])

    #separa as mensagens em treinamento e teste
    quant_treinamento = int(QUANT_TOTAL_MENSAGENS * TRAIN_SIZE_RATIO)
    quant_teste = int(QUANT_TOTAL_MENSAGENS * TEST_SIZE_RATIO)

    #adiciona os tokens e a toxicidade de cada mensagem
    for msg in mensagens_json[:quant_treinamento]:
        msg['tokens'] = msg['content'].lower().split()
        msg['all_toxic_occurrences'] = []
        msg['toxic_tuple'] = (None, 0)

        # for tok in msg['tokens']:
        #     for termo in profanity_list:
        #         if tok.lower() == termo['text'].lower():
        #             msg['all_toxic_occurrences'].append((termo['text'], termo['severity_rating']))

        if msg['all_toxic_occurrences']:
            msg['toxic_tuple'] = max(msg['all_toxic_occurrences'], key=lambda x: x[1])

    #adiciona os tokens e a toxicidade de cada mensagem
    for msg in mensagens_json:
        if 'toxic_tuple' not in msg:
            msg['toxic_tuple'] = (None, 0)

    #pega o conteudo e toxicidades das mensagens obtidas
    contents = [msg['content'] for msg in mensagens_json]
    toxicities = [msg['toxic_tuple'][1] if msg['toxic_tuple'] else 0 for msg in mensagens_json]

    #carrega o dataset de novos termos (si2)
    new_terms_df = pd.read_csv('toxicity_en-newterms.csv')
    new_contents = new_terms_df['text'].tolist()
    new_toxicities = [1 if label == 'Toxic' else 0 for label in new_terms_df['is_toxic']]

    #combina os datasets de treino
    combined_contents = contents + new_contents
    combined_toxicities = toxicities + new_toxicities

    #vê o tamanho do datset
    print(f"Total dataset size: {len(combined_contents)} samples")

    #tokeniza e padroniza as mensagens
    ##########
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(combined_contents)
    #
    X_sequences = tokenizer.texts_to_sequences(combined_contents)
    X_padded = pad_sequences(X_sequences, maxlen=MAX_LENGTH, padding='post', truncating='post')
    ##########

    #pega os dados X e Y pro algoritmo de treino
    small_sample_size = int(len(X_padded) * 0.1)
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded[:small_sample_size], combined_toxicities[:small_sample_size], 
        train_size=0.8, test_size=0.2, random_state=SEED
    )

    #checa se existe mensagens duplicadas (pra deixar maius rapido)
    train_set = set(tuple(row) for row in X_train)
    test_set = set(tuple(row) for row in X_test)
    overlap = train_set.intersection(test_set)
    print(f"Duplicate samples in train/test split: {len(overlap)}")

    #converte os dados para usar no tensorflow
    X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float32)




    #modelo da AI
    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=128, input_length=MAX_LENGTH),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.4),
        Bidirectional(LSTM(32)),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])




    #compila o modelo
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])

    #treina o modelo
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

    history = model.fit(X_train_tensor, y_train_tensor, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=[early_stopping])


    #printa o resultado do treinamento
    for epoch in range(len(history.history['loss'])):
        print(f"Epoch {epoch + 1}/{len(history.history['loss'])}")
        print(f" - loss: {history.history['loss'][epoch]:.4f} - accuracy: {history.history['accuracy'][epoch]:.4f}")
        print(f" - val_loss: {history.history['val_loss'][epoch]:.4f} - val_accuracy: {history.history['val_accuracy'][epoch]:.4f}")

    print(f'#####\n')

    #acha a acc
    loss, accuracy = model.evaluate(X_test_tensor, y_test_tensor)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    #tenta achar as msgs toxicas
    y_pred = model.predict(X_test_tensor)
    y_pred = (y_pred > 0.5).astype(int).flatten()




    # Identify and print the messages that the AI thinks are toxic
    toxic_messages = [combined_contents[i] for i in range(len(y_test)) if y_pred[i] == 1]


    with open("out_foundmessages.txt", "w") as f:
        # Additional checks for debugging
        f.write(f"Number of combined contents: {len(combined_contents)}\n")
        f.write(f"Number of combined toxicities: {len(combined_toxicities)}\n")
        f.write(f"Sample tokenized and padded sequence: {X_padded[0]}\n")
        f.write(f"Training data shape: {X_train_tensor.shape}\n")
        f.write(f"Validation data shape: {X_test_tensor.shape}\n")
        f.write(f"Number of toxic messages found: {len(toxic_messages)}\n")
        
        for message in toxic_messages:
            f.write(f"{message}\n")