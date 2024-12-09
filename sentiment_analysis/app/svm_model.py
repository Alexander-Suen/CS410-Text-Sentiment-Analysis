import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, pipeline
import torch
from torch.nn.functional import softmax
import gensim
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from datasets import load_dataset, Dataset 
import evaluate  
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report


class SVMSentiment:
    def __init__(self, verbose=False, load_pretrained=True):
        self.model_dir = Path("models/svm_sentiment")
        print(f"Looking for model in: {self.model_dir.absolute()}")
        self.is_trained = False
        
        if load_pretrained and self.model_dir.exists():
            print(f"Found model directory at {self.model_dir}") if verbose else None
            try:
                self.is_trained = True
                print("Successfully loaded pre-trained model and tokenizer") if verbose else None
                return  
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                print("Falling back to default model")
        else:
            print(f"No saved model found at {self.model_dir}, using default model") if verbose else None
        
        
        
        print("Loading dataset...") if verbose else None
        self.dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty")
        
        print("Converting to DataFrame...") if verbose else None
        self.df = pd.DataFrame(self.dataset['full'])
        print(self.df.columns)
        
        print("Splitting data...") if verbose else None
        self.train_data, self.test_data = self.tt_split(verbose)
        
        self.x_tokenized_train = [[w for w in cleanText(sentence).split(" ") if w != ""] for sentence in self.train_data['text']]
        print(self.x_tokenized_train[0])
#         self.y_tokenized_train = [[w for w in cleanText(sentence).split(" ") if w != ""] for sentence in self.train_data['labels']]
#         print(self.y_tokenized_train[0])
        
        self.x_tokenized_test = [[w for w in cleanText(sentence).split(" ") if w != ""] for sentence in self.test_data['text']]
        print(self.x_tokenized_test[0])
#         self.y_tokenized_test = [[w for w in cleanText(sentence).split(" ") if w != ""] for sentence in self.test_data['labels']]
#         print(self.y_tokenized_test[0])
        
        self.word2v = gensim.models.Word2Vec(self.x_tokenized_train,vector_size=200)
        
        self.x_train_vec = self.sentence_vector(self.x_tokenized_train)
        self.x_test_vec = self.sentence_vector(self.x_tokenized_test)
        self.svm = SVC(kernel='linear', probability=True)
        self.svm.fit(self.x_train_vec, self.train_data['labels'])
        
        y_pred = self.svm.predict(self.x_test_vec)
        mets = self.compute_metrics(y_pred, self.test_data['labels'])
        print(f"Accuracy: {accuracy}")
        print(f"F1: {f1}")
        

        

    # def preprocess_function(self, examples):
    #     return self.tokenizer(examples["text"], truncation=True)
    
   
    def compute_metrics(self, y_true, y_pred):
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='binary')
        
        # Print detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        return {"accuracy": accuracy, "f1": f1}
    

    def tt_split(self, verbose=False):
        tqdm.pandas(desc="Creating sentiment labels", disable=not verbose)
        self.df['sentiment'] = self.df['rating'].progress_apply(lambda x: 1 if x >= 4 else 0)

        # Select features and target
        X = self.df['text']
        y = self.df['sentiment']

        if len(X) > 11000:
            if verbose:
                print(f"Reducing dataset from {len(X)} to 11000 points...")
            indices = np.random.choice(len(X), 11000, replace=False)
            X = X.iloc[indices]
            y = y.iloc[indices]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=1000, 
            random_state=42,
            stratify=y
        )

        if len(X_train) > 10000:
            X_train = X_train[:10000]
            y_train = y_train[:10000]

        if verbose:
            print(f"Creating training set with {len(X_train)} samples and test set with {len(X_test)} samples...")
        train_texts = list(tqdm(X_train, desc="Processing train texts", disable=not verbose))
        train_labels = list(tqdm(y_train, desc="Processing train labels", disable=not verbose))
        test_texts = list(tqdm(X_test, desc="Processing test texts", disable=not verbose))
        test_labels = list(tqdm(y_test, desc="Processing test labels", disable=not verbose))

        train_dataset = {
            'text': train_texts,
            'labels': train_labels
        }

        test_dataset = {
            'text': test_texts,
            'labels': test_labels
        }

        return train_dataset, test_dataset
    
    
    def sentence_vector(self, sentences):
        vecs = []
        for sentence in sentences:
            sv = [self.word2v.wv[word] for word in sentence if word in self.word2v.wv]
            if sv:
                vecs.append(np.mean(sv, axis=0))
            else:
                vecs.append(np.zeros(self.word2v.vector_size))
        return np.array(vecs)


def cleanText(text):
    cleaned = re.sub("[^a-zA-Z0-9']"," ",text)
    lowered = cleaned.lower()
    return lowered.strip()