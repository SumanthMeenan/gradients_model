from typing import Tuple
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

MAX_SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 128

def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """loads the test file and extracts all functions/derivatives"""
    data = open(file_path, "r").readlines()
    functions, derivatives = zip(*[line.strip().split("=") for line in data])
    return functions, derivatives

def prepare_data(functions: Tuple[str], derivatives: Tuple[str]):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(functions)
    tokenizer.fit_on_texts(derivatives)
    input_sequences = tokenizer.texts_to_sequences(functions)
    input_sequences = pad_sequences(input_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    output_sequences = tokenizer.texts_to_sequences(derivatives)
    output_sequences = pad_sequences(output_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    vocab_size = len(tokenizer.word_index) + 1
    output_sequences = to_categorical(output_sequences, num_classes=vocab_size)  # One-hot encode the output sequences
    return input_sequences, output_sequences, vocab_size, tokenizer

def create_model(vocab_size: int):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(LSTM(128))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, input_sequences, output_sequences):
    model.fit(input_sequences, output_sequences, epochs=10, batch_size=32)

def predict(functions: str, tokenizer, model):
    input_sequences = tokenizer.texts_to_sequences([functions])
    input_sequences = pad_sequences(input_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    predicted_sequence = model.predict(input_sequences)
    predicted_derivative = tokenizer.sequences_to_texts(np.argmax(predicted_sequence, axis=-1))
    return predicted_derivative

def score(true_derivative: str, predicted_derivative: str) -> int:
    """binary scoring function for model evaluation"""
    return int(true_derivative == predicted_derivative)


def main(filepath: str = "train.txt"):
    """load, train, inference, and evaluate"""
    functions, true_derivatives = load_file(filepath)
    print(2)
    input_sequences, output_sequences, vocab_size, tokenizer = prepare_data(functions, true_derivatives)
    model = create_model(vocab_size)
    train_model(model, input_sequences, output_sequences)
    predicted_derivatives = [predict(f, tokenizer, model) for f in functions]
    scores = [score(td, pd) for td, pd in zip(true_derivatives, predicted_derivatives)]
    print(np.mean(scores))

if __name__ == "__main__":
    main()
