from src.data_processing import deep_learning_processing, Sales_to_sentences
from src.model import bidirectional_LSTM
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == "__main__": 

    # Load the data
    sales = pd.read_csv("receipts.csv")
    articles = pd.read_csv("articles.csv") 

    sub_sales = sales.head(1000)
    # Preprocess the data
    sentences = Sales_to_sentences(sub_sales)
    transactions = sentences.sales_to_sentences()

    # Create the input and output for the model
    dl_processing = deep_learning_processing(transactions)
    xs, ys = dl_processing.get_xs_ys()

    # retrieve the maximum sequence length and the vocabulary size
    max_sequence_len = dl_processing.max_sequence_len 
    total_words = len(dl_processing.tokenizer.word_index) + 1

    # train the model
    
    model = bidirectional_LSTM(total_words, max_sequence_len)
    model = model.create_model()
    model.fit(xs, ys, epochs=30, verbose=1, batch_size=32)

    # retrieve the predictions
    seed_text = "1412486 3026575"

    recommendations = dl_processing.recommendation_retrieval(seed_text, 10, model, articles)
    recommendations.to_csv("recommendations.csv", index=True)

