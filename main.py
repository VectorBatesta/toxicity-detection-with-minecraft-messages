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

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

if __name__ == "__main__":
    set_seeds()

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    with open("clean.json", "r", encoding="utf8") as f:
        mensagens_json = json.load(f)

    with open("profanity_en.csv", "r", encoding="utf8") as f:
        profanity_list = list(csv.DictReader(f))
    for termo in profanity_list:
        termo['severity_rating'] = float(termo['severity_rating'])

    quant_total_mensagens = 1000
    quant_treinamento = int(quant_total_mensagens * 0.7)
    quant_teste = int(quant_total_mensagens * 0.3)

    for msg in mensagens_json[:quant_treinamento]:
        msg['tokens'] = msg['content'].lower().split()
        msg['all_toxic_occurrences'] = []
        msg['toxic_tuple'] = (None, 0)

        for tok in msg['tokens']:
            for termo in profanity_list:
                if tok.lower() == termo['text'].lower():
                    msg['all_toxic_occurrences'].append((termo['text'], termo['severity_rating']))

        if msg['all_toxic_occurrences']:
            msg['toxic_tuple'] = max(msg['all_toxic_occurrences'], key=lambda x: x[1])

    for msg in mensagens_json:
        if 'toxic_tuple' not in msg:
            msg['toxic_tuple'] = (None, 0)

    contents = [msg['content'] for msg in mensagens_json]
    toxicities = [msg['toxic_tuple'][1] if msg['toxic_tuple'] else 0 for msg in mensagens_json]

    new_terms_df = pd.read_csv('toxicity_en-newterms.csv')
    new_contents = new_terms_df['text'].tolist()
    new_toxicities = [1 if label == 'Toxic' else 0 for label in new_terms_df['is_toxic']]

    combined_contents = contents + new_contents
    combined_toxicities = toxicities + new_toxicities

    print(f"Total dataset size: {len(combined_contents)} samples")

    vocab_size = 10000
    max_length = 50

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(combined_contents)

    X_sequences = tokenizer.texts_to_sequences(combined_contents)
    X_padded = pad_sequences(X_sequences, maxlen=max_length, padding='post', truncating='post')

    small_sample_size = int(len(X_padded) * 0.1)
    X_train, X_test, y_train, y_test = train_test_split(
        X_padded[:small_sample_size], combined_toxicities[:small_sample_size], 
        train_size=0.8, test_size=0.2, random_state=42
    )

    train_set = set(tuple(row) for row in X_train)
    test_set = set(tuple(row) for row in X_test)
    overlap = train_set.intersection(test_set)
    print(f"Duplicate samples in train/test split: {len(overlap)}")

    X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float32)

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.4),
        Bidirectional(LSTM(32)),
        Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(X_train_tensor, y_train_tensor, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    loss, accuracy = model.evaluate(X_test_tensor, y_test_tensor)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
