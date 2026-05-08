Twitter Sentiment Analysis – Project Report
Twitter Sentiment Analysis – Project Report
1. Problem Statement
The objective of this project is to build an NLP system capable of analyzing tweets and classifying them into Positive, Negative, Neutral, or Irrelevant sentiments. The project automates sentiment detection from social media text for applications such as brand monitoring, customer feedback analysis, political campaigns, and market research.
2. Dataset Description.
Dataset Source:
LINK OF DATASET:
Features:
- Tweet/Text
- Sentiment Label
Sentiment Classes:
0 - Irrelevant
1 - Negative
2 - Neutral
3 - Positive
3. Data Preprocessing
The following preprocessing steps were applied:
- Lowercasing
- URL removal
- Mention removal
- Hashtag removal
- Number removal
- Punctuation removal
- Tokenization
- Stopword removal
- Lemmatization
4. Methods Used
A. TF-IDF + Logistic Regression
- TF-IDF vectorization converts text into numerical features.
- Logistic Regression is used for sentiment classification.
Advantages:
- Fast and lightweight
- Effective on sparse text data
- Easy deployment
B. BiLSTM Deep Learning Model
Architecture:
- Embedding Layer
- SpatialDropout1D
- Bidirectional LSTM
- GlobalMaxPooling1D
- Dense + Dropout Layers
- Softmax Output Layer
Benefits:
- Better contextual understanding
- Improved sequence learning
- More accurate for complex sentiment patterns
