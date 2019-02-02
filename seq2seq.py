from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences
import re
import helper
import numpy as np


with open("conversations.txt", "r", encoding="utf-8") as f:
    sentences_list = f.readlines()
sentences = [re.sub("^- ", "", i) for i in sentences_list if i != ""]
dataset, id2word, word2id, MAX_LEN = helper.prepare_for_dataset(sentences)

HIDDEN_SIZE = 256
DECODER_SIZE = 256
EMBEDDING_SIZE = 100
VOCABULARY_SIZE = len(id2word)
WORD_DIM = 300
EPOCHS = 5
BATCH_SIZE = 4


def decode(num):
    result = [0] * VOCABULARY_SIZE
    result[num] = 1
    return result


# Define an input sequence and process it.
encoder_inputs = Input(shape=(MAX_LEN,))
embeddings = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBEDDING_SIZE)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(HIDDEN_SIZE, return_state=True)(embeddings)
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(MAX_LEN,))
decoder_embeddings_inputs = Embedding(VOCABULARY_SIZE, EMBEDDING_SIZE)(decoder_inputs)
decoder_lstm = LSTM(DECODER_SIZE, return_sequences=True)(decoder_embeddings_inputs, initial_state=encoder_states)
decoder_dense = Dense(VOCABULARY_SIZE, activation='softmax')(decoder_lstm)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_dense)

# Compile & run training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!
print(model.summary())
encoder_input_data = pad_sequences([i[0] for i in dataset], maxlen=MAX_LEN, padding="post")
decoder_target_data = pad_sequences([[decode(ind) for ind in i[1]] for i in dataset], maxlen=MAX_LEN, padding="post")
decoder_input_data = pad_sequences([[0]+i[1][:-1] for i in dataset], maxlen=MAX_LEN, padding="post")

model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.2)

# INFERENCE
# ------------------------------------------------------------

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(MAX_LEN,))
decoder_state_input_c = Input(shape=(MAX_LEN,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, EMBEDDING_SIZE))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, 0] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = id2word[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > MAX_LEN):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, VOCABULARY_SIZE))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence