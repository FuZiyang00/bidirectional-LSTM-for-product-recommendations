![Alt text](https://github.com/FuZiyang00/bidirectional-LSTM-for-product-recommendations/blob/main/wall_paper.png)

# Bidirectional LSTM to generate recommendations 
The aim of this project is demonstrate how NLP techniques can be applied to a classic business problem like *products recommendations* .
In such sense, we can sense users' transactions using another lense: treat them like sentences. 

## Overview

In this project, we treat each user's transaction history as a "sentence" where each product purchased is a "word". This allows us to apply Natural Language Processing (NLP) techniques, specifically a Bidirectional Long Short-Term Memory (LSTM) model, to generate product recommendations.

## Data

The data used in this project consists of transaction records, with each record representing a product purchased by a user. The data is preprocessed into a format suitable for training the LSTM model.

The employed data presents the following structure: 
| STORE_ID|DATE| TIMESTAMP| REGISTER_ID| RECEIPT_ID|PIECES|UNIT_PRICE
| :-------------:|:-------------:| :-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
|88| 2023-03-03|08:03:13|19|2|310424|1|12.90|

## Model

The Bidirectional LSTM model is a type of recurrent neural network that is capable of learning patterns in sequences of data. In this project, the model learns patterns in the sequence of products purchased by users, and uses this knowledge to predict what products a user might purchase next.

## Project structure 
```
project-root/
│
├── src/
│ ├── data_processing.py
│ └── model.py
|
│── main.py
├── README.md
```

