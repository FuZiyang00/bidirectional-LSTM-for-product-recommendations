from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class bidirectional_LSTM:
    def __init__(self, total_words, max_sequence_len):
        self.total_words = total_words
        self.max_sequence_len = max_sequence_len

    def create_model(self):
        """
        This method creates a deep learning model with a bidirectional LSTM layer.
        """
        model = Sequential()
        model.add(Embedding(self.total_words, 100, input_length=self.max_sequence_len-1))
        model.add(Bidirectional(LSTM(150)))
        model.add(Dense(self.total_words, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model 
    

        
