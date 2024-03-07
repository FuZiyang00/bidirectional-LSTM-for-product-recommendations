import pandas as pd
import numpy as np
import string

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Sales_to_sentences: 

    def __init__(self, sales):
        self.sales = sales 

    def sales_to_sentences(self):
        """
        This method takes a dataframe of sales and groups the items belonging to the same receipt, where each 
        observation is a sentence of the form product_1 product_2  ... product_n that represents 
        a single transaction. Grouping criteria item Ids presenting the same store_ID, date, time_stamp, register_ID, 
        and receipt_ID (this number is not unique) are considered items belonging to the same transaction.
        """
        transactions = self.sales.groupby(['store_ID', 'date', 'time_stamp', 'register_ID', 'receipt_ID']) \
            .apply(lambda group: list(group['item_ID'].values)) \
            .reset_index()
    
        transactions.drop(['store_ID', 'date', 'time_stamp', 'register_ID', 'receipt_ID'], axis = 1, inplace=True) # drop the unnecessary columns
        transactions['transaction_ID'] = np.arange(len(transactions)) + 1 # assign an unique ID to each transaction
        transactions = transactions.rename(columns={0: 'basket'})
        transactions["transaction_length"] = transactions['basket'].apply(len)

        #lambda function to remove punctuation
        remove_punctuation = lambda x: ''.join(char for char in x if char not in string.punctuation)

        # transform the list of items into a string
        transactions['basket'] = transactions['basket'].apply(lambda x: ', '.join(map(str, x))).apply(remove_punctuation)

        return transactions
    

class deep_learning_processing: 

    def __init__(self, transactions):
        self.transactions = transactions
        self.tokenizer = Tokenizer(oov_token='<oov>')
        self.max_sequence_len = None

    def get_xs_ys(self):
        """
        This method takes a dataframe of transactions and returns the input and output for the deep learning model.
        """
        self.tokenizer.fit_on_texts(self.transactions['basket'])
        total_words = len(self.tokenizer.word_index) + 1
        
        # generate ngrams 
        input_sequences = []
        for line in self.transactions['basket']:
            token_list = self.tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)
        
        # pad sequences
        self.max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=self.max_sequence_len, padding='pre'))

        xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
        ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

        return xs, ys
    

    def recommendation_retrieval(self, seed_text, next_words, model, df):

        for _ in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=self.max_sequence_len-1, padding='pre')
            predicted_probs = model.predict(token_list, verbose=0)[0]  # Get the predicted probabilities for next word
            predicted_index = np.argmax(predicted_probs)  # Get the index of the word with highest probability
            output_word = ""  # Initialize output word

            for word, index in self.tokenizer.word_index.items():  # Find the word corresponding to the index
                if index == predicted_index:
                    output_word = word
                    break
            seed_text += " " + output_word  # Append the predicted word to the seed text
        
        l = [int (i) for i in seed_text.split(' ')]

        output = pd.DataFrame(columns=df.columns)

        for i in range(len(l)): 
            selected_row = df[df['item_ID'] == l[i]]
            if not selected_row.empty:
                output = pd.concat([output, selected_row])

        output.reset_index(drop=True, inplace=True)
        return output
    


    
