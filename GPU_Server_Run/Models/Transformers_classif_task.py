#!/usr/bin/env python

# -------------INCIPIT------------------
# The goal is to perform Sentiment Analysis with two configurations of the transformer BERT: DistilBERT and FinBERT. The task is classification.
# Specifically the pre-trained models will be downloaded from the respectives HuggingFace repos and will be fine-tuned on the other Input Data like the financial phrasebank.
# ---------------------------------------



import time
start_time = time.time()



import pandas as pd
import os
from datetime import date
import json
import nltk
nltk.download('punkt')
nltk.download("stopwords")
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.corpus import stopwords
import transformers
from datasets import Dataset
import numpy as np
from datasets import load_metric
from transformers import TrainingArguments, Trainer
import evaluate
import shutil
import tempfile
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
from huggingface_hub import login, logout
from transformers import logging


def HF_dataset_splitting(df):
    df = Dataset.from_pandas(df)
    train_size = int(df.shape[0] * 0.8)
    test_size = df.shape[0] - train_size
    train_dataset = df.shuffle(seed=42).select([i for i in list(range(train_size))])
    test_dataset = df.shuffle(seed=42).select([i for i in list(range(test_size))])
    return df, train_dataset, test_dataset



def HF_tokenizer(model):
    model = str(model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    return tokenizer



def HF_padding(tokenizer):
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
    return data_collator


def HF_load_model(model_name, num_target_labels):
    model_name = str(model_name)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_target_labels)
    return model



def compute_metrics(eval_pred, average):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    f1 = f1_score(y_true=labels, y_pred=predictions, average=average)
    
    return {"accuracy": accuracy, "f1": f1}



def ConfrontBuilder(confront, model_name, accuracy, f1, exec_time):
    confront.loc[:, model_name] = accuracy, f1, exec_time
    return confront




current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
print(parent_directory)


# Load the dataset already pre-processed in main.py
df4 = pd.read_csv(parent_directory + '/Input_Data/Final_Dataset.csv')



token = "Type here your HF API Token"

login(token) # non-blocking login


logging.set_verbosity_error()

model_name = ["distilbert-base-uncased", "yiyanghkust/finbert-tone"]

df4_HF, train_dataset, test_dataset = HF_dataset_splitting(df4)


def preprocess_function(examples):
	target = 'sentence'
	return tokenizer(examples[target], truncation=True)


# Create a temporary directory for output
repo_name = [tempfile.mkdtemp(), tempfile.mkdtemp()]


confusion_matrices = []
precisions = []
recalls = []


confront = pd.DataFrame(index=['Accuracy', 'F1', 'ExecTime(sec)'])
for i in range(0,len(model_name)):
    tokenizer = HF_tokenizer(model=model_name[i])
    
    
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = HF_padding(tokenizer)

    model = HF_load_model(model_name= model_name[i], num_target_labels= 3)

    training_args = TrainingArguments(
        output_dir=repo_name[i], # saves parameters and other info into a heavy folder
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        save_strategy="epoch",
        push_to_hub=True,
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=lambda p: compute_metrics(p, average="micro"),  # Pass "micro" as the average parameter
    )


    t1 = time.time()
    trainer.train()
    t2 = time.time()

    evaluation_result = trainer.evaluate()
    accuracy = evaluation_result['eval_accuracy']
    f1 = evaluation_result['eval_f1']

    exec_time = t2 - t1
    
    # After training is complete, you can delete the temporary directory if you don't want to keep the outputs
    shutil.rmtree(repo_name[i])
	
    print(str(model_name[i]))
    print("Accuracy: " + str(accuracy))
    print("F1: " + str(f1))
    print("Execuation time: " + str(exec_time))
    confront = ConfrontBuilder(confront, model_name[i], accuracy, f1, exec_time)
    confront.to_csv(parent_directory + f'/Output/Transformers/model_{i}_performance.csv')
    
    
    # After training is complete, predict labels on the test dataset
    predictions = trainer.predict(tokenized_test)
    predicted_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = tokenized_test["label"]

    # Calculate the confusion matrix, precision, and recall
    confusion_matrix_i = confusion_matrix(true_labels, predicted_labels)
    precision_i = precision_score(true_labels, predicted_labels, average='micro')
    recall_i = recall_score(true_labels, predicted_labels, average='micro')

    # Append the results to your lists
    confusion_matrices.append(confusion_matrix_i)
    precisions.append(precision_i)
    recalls.append(recall_i)


confront.to_csv(parent_directory + '/Output/Transformers/models_performance.csv') 


results_df = pd.DataFrame({
    "Model Name": model_name,
    "Confusion Matrix": confusion_matrices,
    "Precision": precisions,
    "Recall": recalls
})

results_df.to_csv(parent_directory + '/Output/Transformers/sentiment_analysis_metrics.csv', index=False)


end_time = time.time()



logout() # logout completely from HuggingFace


run_time = end_time - start_time
print("The script running takes " + str(round(run_time,0)) + " seconds to end")

