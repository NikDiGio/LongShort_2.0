#!/usr/bin/env python3


# -------------INCIPIT------------------
# Here the task is for regression therefore we predict the sentiment scores associated to all the headlines web-scraped from the internet to combine these with the other features used for regression. 
# This Python script contains: 
#	- the code to text-embed the sentences using a pre-trained Word2vec model from Gensim pre-trained on Glove 6B and fine-tuned on the other Input Data like the financial phrasebank dataset;
#	- the code to perform sentiment analysis with a CNN on the sequences of numbers obtained through the previous step.
# ---------------------------------------

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # A tensorfow compatibility problem with the CUDA configuration of the server pushed me to run the script on CPU 

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Input, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from transformers import AutoModel, AutoTokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import matplotlib.pyplot as plt
import pickle
import time

from huggingface_hub import login, logout
from transformers import logging
from tensorflow.keras.utils import to_categorical
import re
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
import gensim.downloader as api
from scipy import stats

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
print(parent_directory)


df = pd.read_csv(parent_directory + '/Input_Data/Final_Dataset.csv')

###### TEST ######
df = df.iloc[-3000:]
##################

df2 = pd.read_csv(parent_directory + '/Input_Data/WS_Final_Dataset.csv', index_col = 0) # for testing (unlabelled). WS -> Web Scraping


# Define the preprocessing function
def preprocess_text(text):
    # Remove non-alphanumeric characters, punctuation, and special characters
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    # Convert text to lowercase
    text = text.lower()
    return text



# left padding the text embedding
def left_pad_sequences_with_torch(sequences, max_sequence_length):
    padded_sequences = []
    for sequence in sequences:
        pad_length = max_sequence_length - len(sequence)
        sequence = torch.tensor(sequence, dtype=torch.float32) # sequences are np.arrays so conversion in torch.tensor is needed
        if pad_length > 0:
            pad = torch.zeros(pad_length, sequence.size(1), dtype=sequence.dtype)
            padded_sequence = torch.cat((pad, sequence), dim=0)
        else:
            padded_sequence = sequence
        padded_sequences.append(padded_sequence)
        
    return torch.stack(padded_sequences)



# -------------A BIT MORE OF PREPROCESSING------------------

# Filter out the special token from each sequence
filtered_sequences = [re.sub(r'</s>', '', sequence) for sequence in df['sentence']]
filtered_sequences_2 = [re.sub(r'</s>', '', sequence) for sequence in df2['sentence']]

# Look at the differences:
# before:
print(df['sentence'])


df['sentence'] = filtered_sequences
df2['sentence'] = filtered_sequences_2


# Apply the preprocessing function to the 'text' column
df['sentence'] = df['sentence'].apply(preprocess_text)
df2['sentence'] = df2['sentence'].apply(preprocess_text)
# after:
print(df['sentence'])

# Check if after removing the </s> tokens and spaces there are blanks in the sentence column
empty_cells_count = df['sentence'].str.strip().eq('').sum()
print(f"The 'sentence' column has {empty_cells_count} empty cells.")

# Filter out rows where the 'sentence' column is empty
df = df.loc[df['sentence'].str.strip().eq('') == False]
df2 = df2.loc[df2['sentence'].str.strip().eq('') == False]

# ------------------------------------------------------------


# Data splitting

test_size = df2.shape[0]
train_size = df.shape[0]
val_size = int(train_size * 0.20)

print("Test size ", test_size)
print("Train size ", train_size)
print("Val size ", val_size)

X = df['sentence']  # Features (text data)
Y = df['label']  # Labels (sentiment labels)

# No Split the data into training and test sets (e.g., 80% training, 20% test) since the WS headlines are used for testing


# Optionally, you can further split the training data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.20, random_state=42)

# Web scraped headlines as test features
X_test = df2['sentence']

print(X_train)

num_classes = 3

# one-hot encode the labels using TensorFlow and Keras
Y_train = to_categorical(Y_train, num_classes)
Y_val = to_categorical(Y_val, num_classes)


# ----------------------WORD2VEC------------------------------

# The WORD2VEC has to be fine tuned on the sentences 

# Record the start time for word2vec model
start_time_word2vec = time.time()

full_sentences = pd.concat([X_train, X_val, X_test])

# Tokenize the sentences in your training data
tokenized_sentences = [word_tokenize(sentence) for sentence in full_sentences]

# First build a custom Word2Vec on your data
embedding_dim= 300

model = Word2Vec(vector_size=embedding_dim, min_count=1)
model.build_vocab(tokenized_sentences)
total_examples = model.corpus_count

print('total_examples: ',total_examples)

# Save the vocab of your dataset
vocab = list(model.wv.key_to_index.keys())

print("Fine tuning vocab len",len(vocab))

# Given the file size limit of GitHub of 100MB at most I needed to slice the txt file. 
# Here i load all of them as a single one saved as a temporary file that will be removed after having its course

# Initialize a variable to store combined content
combined_content = ''

directory_path = parent_directory + '/Input_Data/glove.6B/'
file_list = [file for file in os.listdir(directory_path) if file.startswith('glove.6B.300d') and file.endswith('.txt')]

for file_name in file_list:
    with open(directory_path + file_name, 'r') as file:
        content = file.read()
        # Append the content of each file to the combined content variable
        combined_content += content

# Use a temporary file to store the combined content
with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:    
    temp_file.write(combined_content)
    temp_file_path = temp_file.name  # Retrieve the temporary file path
    print('temp_file_path \n', temp_file_path)
    
# Count the number of rows (lines) in the temporary file
with open(temp_file_path, 'r') as temp_file:
    content = temp_file.readlines()
    total_rows = len(content)
    print(f"Total number of rows: {total_rows}")
    
    
# Load the pre-trained model
model_2 = KeyedVectors.load_word2vec_format(temp_file_path, binary=False)


# Add the pre-trained model vocabulary
model.build_vocab([list(model_2.key_to_index.keys())], update=True)


print("Pre trained vocab len",len(list(model_2.key_to_index.keys())))

# Combine the two models
# note: if a word doesn't exist in the pre-trained vocabulary then it is left as is in the original model
model.wv.load_word2vec_format(temp_file_path, binary=False)

# -------------------------------Remove temp file----------------------------------------------------
# At the end of the process, delete the temporary file to avoid getting file size troubles in pushing changes to GitHub
os.remove(temp_file_path)
# ------------------------------------------------------------------------------------------

# Fine tune the pre trained Word2Vec with my dataset
model.train(tokenized_sentences, total_examples=total_examples, epochs=5)

# Save the len of vocab of your dataset
vocab = len(list(model.wv.key_to_index.keys()))


# Get vocabulary words
vocab_words = model.wv.index_to_key

# Get corresponding indices
vocab_indices = [model.wv.key_to_index.get(word) for word in vocab_words]


# Combine vocabulary words, indices, and vectors into a DataFrame
vocab_df = pd.DataFrame({'Word': vocab_words, 'Index': vocab_indices})


vocab_df.to_csv(parent_directory + '/Output/CNN/vocabulary_with_indices_new_regress.csv')


# Lists to store your sequences
embedded_sequences_X_train = []
embedded_sequences_X_test = []
embedded_sequences_X_val = []


# Process your training data
for sentence in X_train:
    tokens_X_train = word_tokenize(sentence)
    embedded_sequence = [
        model.wv[token]
        for token in tokens_X_train
    ]
    embedded_sequences_X_train.append(embedded_sequence)

# Process your test data
for sentence in X_test:
    tokens_X_test = word_tokenize(sentence)
    embedded_sequence = [
        model.wv[token]
        for token in tokens_X_test
    ]
    embedded_sequences_X_test.append(embedded_sequence)

# Process your validation data
for sentence in X_val:
    tokens_X_val = word_tokenize(sentence)
    embedded_sequence = [
        model.wv[token]
        for token in tokens_X_val
    ]
    embedded_sequences_X_val.append(embedded_sequence)

# Determine max_sequence_length after text embedding
max_sequence_length_train = max(len(seq) for seq in embedded_sequences_X_train)
max_sequence_length_test = max(len(seq) for seq in embedded_sequences_X_test)
max_sequence_length_val = max(len(seq) for seq in embedded_sequences_X_val)

# The overall max_sequence_length is the maximum among the three sets
max_sequence_length = max(max_sequence_length_train, max_sequence_length_test, max_sequence_length_val)
print("Maximum Sequence Length:", max_sequence_length)

# Converting lists of text embeddings to a NumPy array to get a more efficient code
embedded_sequences_X_train = np.array(embedded_sequences_X_train, dtype=object) # creating an ndarray from ragged nested sequences
embedded_sequences_X_val = np.array(embedded_sequences_X_val, dtype=object) # creating an ndarray from ragged nested sequences
embedded_sequences_X_test = np.array(embedded_sequences_X_test, dtype=object) # creating an ndarray from ragged nested sequences


# Left-pad the sequences for training, validation, and test sets
left_padded_sequences_X_train = left_pad_sequences_with_torch(embedded_sequences_X_train, max_sequence_length)
left_padded_sequences_X_val = left_pad_sequences_with_torch(embedded_sequences_X_val, max_sequence_length)
left_padded_sequences_X_test = left_pad_sequences_with_torch(embedded_sequences_X_test, max_sequence_length)


print("Left-padded sequences_X_train shape: ", left_padded_sequences_X_train.shape)
print("Left-padded sequences_X_val shape: ", left_padded_sequences_X_val.shape)
print("Left-padded sequences_X_test shape: ", left_padded_sequences_X_test.shape)


# Convert PyTorch tensors to TensorFlow tensors
embedded_sequences_X_train = tf.convert_to_tensor(left_padded_sequences_X_train.numpy(), dtype=tf.float32)
embedded_sequences_X_val = tf.convert_to_tensor(left_padded_sequences_X_val.numpy(), dtype=tf.float32)
embedded_sequences_X_test = tf.convert_to_tensor(left_padded_sequences_X_test.numpy(), dtype=tf.float32)


# Reshape the TensorFlow tensors
reshaped_embedded_sequences_X_train = tf.reshape(embedded_sequences_X_train, (embedded_sequences_X_train.shape[0], -1))
reshaped_embedded_sequences_X_test = tf.reshape(embedded_sequences_X_test, (embedded_sequences_X_test.shape[0], -1))
reshaped_embedded_sequences_X_val = tf.reshape(embedded_sequences_X_val, (embedded_sequences_X_val.shape[0], -1))


print("embedded_sequences_X_train shape: ", embedded_sequences_X_train.shape)
print("embedded_sequences_X_test shape: ", embedded_sequences_X_test.shape)
print("embedded_sequences_X_val shape: ", embedded_sequences_X_val.shape)
print("reshaped_embedded_sequences_X_train shape: ", reshaped_embedded_sequences_X_train.shape)
print("reshaped_embedded_sequences_X_test shape: ", reshaped_embedded_sequences_X_test.shape)
print("reshaped_embedded_sequences_X_val shape: ", reshaped_embedded_sequences_X_val.shape)


# Record the end time for word2vec model
end_time_word2vec = time.time()


# ----------------------CNN------------------------------


# Calculate the execution time for word2vec model
execution_time_word2vec = end_time_word2vec - start_time_word2vec


# Record the start time for CNN
start_time_cnn = time.time()



# Define the input shape
input_shape = (embedding_dim * max_sequence_length,)


input_layer = Input(shape=input_shape)
embedding_layer = Embedding(input_dim=vocab, output_dim=embedding_dim)(input_layer)


# Apply convolutional layers
num_filters = 128
filter_sizes = [3, 4, 5]
# A filter size of 5 means that the convolutional layer will consider only 5 adjacent words (or tokens) at a time. If you believe that important patterns or features in your data might be distributed across a larger window of words, you can consider using larger filter sizes, such as 7, 9, or even larger, to capture longer-range dependencies in your text.

conv_layers = []
max_pool_size = max_sequence_length - max(filter_sizes) + 1  # Compute a common max_pool_size to avoid shape problems in concatenation

for filter_size in filter_sizes:
    
    conv = Conv1D(num_filters, filter_size, activation='relu')(embedding_layer) 

    print(conv.shape)

    pool = MaxPooling1D(pool_size=max_pool_size)(conv)
    conv_layers.append(pool)



# Concatenate the pooled features
concatenated = tf.keras.layers.Concatenate(axis=-1)(conv_layers)  # Specify the axis correctly
flatten = Flatten()(concatenated)

# Dense layers for classification
dense1 = Dense(64, activation='relu')(flatten)
num_classes = 3
output_layer = Dense(num_classes, activation='softmax')(dense1)  # num_classes is 3 in your case (positive, neutral, negative)

# Create the model
model_cnn = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model_cnn.summary()


num_epochs = 50 # by rule of thumb
batch_size = 16 # A common choice for batch_size is in the range of 32 to 128. Put to 16 due to server memory problems


# Validation and early stopping
early_stopping = EarlyStopping(monitor='accuracy', patience=8, restore_best_weights=True) # if the validation loss does not improve for five consecutive epochs, early stopping will be triggered, and the training process will stop.

# Training
history = model_cnn.fit(reshaped_embedded_sequences_X_train, Y_train, validation_data=(reshaped_embedded_sequences_X_val, Y_val), epochs=num_epochs, batch_size=batch_size, callbacks=[early_stopping])


# Testing
# Make predictions on the test data
Y_pred = model_cnn.predict(reshaped_embedded_sequences_X_test)  # This assumes your model outputs class probabilities (e.g., using softmax)
Y_pred = Y_pred.argmax(axis=1)


# Record the end time for CNN
end_time_cnn = time.time()

# Calculate the execution time for CNN
execution_time_cnn = end_time_cnn - start_time_cnn

data = {
    "Model": ["Word2Vec_pretrained_finetuned", "CNN"],
    "Execution Time (s)": [execution_time_word2vec, execution_time_cnn]
}

df_ET = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df_ET.to_csv(parent_directory + '/Output/CNN/execution_times_regress_pretrained.csv', index=False)



# Assuming you have the 'history' object from training
# Plot training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(parent_directory + '/Output/CNN/accuracy_plot_regress_wv_pretrained.png')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(parent_directory + '/Output/CNN/loss_plot_regress_wv_pretrained.png')
plt.show()

# Remove the time and keep only the date (producing duplicate dates)
df2['date'] = [pd.Timestamp(date).date() for date in list(df2.index)]

# Add the predicted labels corresponding to the model
df2['label'] = Y_pred
    

df2.to_csv(parent_directory + '/Output/CNN/WS_Final_Dataset_Predicted_CNN.csv')

# Take the modal sentiment of all the articles published for each day
daily_sentiment = df2.groupby('date')['label'].apply(lambda x: stats.mode(x)[0][0])

daily_sentiment.to_csv(parent_directory + '/Output/CNN/daily_sentiment_CNN_regress.csv')