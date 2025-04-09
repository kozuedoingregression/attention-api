import numpy as np

import tensorflow as tf
from tensorflow import keras

import pickle
import string
import re

import warnings
warnings.filterwarnings("ignore")

from custom_layers import PositionalEmbedding, TransformerDecoder, TransformerEncoder, MultiHeadAttention

model = tf.keras.models.load_model(
    "model/transformer_de_to_en_model.keras",
    custom_objects={
        "PositionalEmbedding": PositionalEmbedding,
        "TransformerEncoder": TransformerEncoder,
        "TransformerDecoder": TransformerDecoder,
        "MultiHeadAttention": MultiHeadAttention
    },
    compile=False
)

strip_chars = string.punctuation
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")
 
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
        lowercase, f"[{re.escape(strip_chars)}]", "")

max_tokens = 25000
sequence_length = 30

with open("model/source_vocab.pkl", "rb") as f:
    source_vocab = pickle.load(f)

with open("model/target_vocab.pkl", "rb") as f:
    target_vocab = pickle.load(f)

# tokenize the data
source_vectorization = keras.layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=sequence_length,
)
target_vectorization = keras.layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=sequence_length + 1, 
    standardize=custom_standardization,
)


source_vectorization.set_vocabulary(source_vocab)
target_vectorization.set_vocabulary(target_vocab)

# For inference
target_index_lookup = dict(enumerate(target_vectorization.get_vocabulary()))
max_decoded_sentence_length = 30

def decode_sequence(input_sentence: str) -> str:
    tokenized_input_sentence = source_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization([decoded_sentence])[:, :-1]
        predictions = model([tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = target_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sentence.replace("[start]", "").replace("[end]", "").strip()

print(decode_sequence("ich bin klug"))
print(decode_sequence("sie ist klug"))
print(decode_sequence("meine bruder spielt klavier"))
print(decode_sequence("er ist kellner "))


