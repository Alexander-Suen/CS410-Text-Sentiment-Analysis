import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, pipeline
import torch
from torch.nn.functional import softmax
import numpy as np
from datasets import load_dataset, load_metric
from sklearn.model_selection import train_test_split
# sentiment_pipeline = pipeline("sentiment-analysis")


class SentimentAnalysisModel:
    def __init__(self, verbose=False):
        print("Initializing tokenizer...") if verbose else None
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        
        print("Initializing model...") if verbose else None
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        
        print("Loading dataset...") if verbose else None
        self.dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty")
        
        print("Converting to DataFrame...") if verbose else None
        self.df = pd.DataFrame(self.dataset['full'])
        
        print("Splitting data...") if verbose else None
        self.train_data, self.test_data = self.tt_split(verbose)
        

    def preprocess_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True)
    
    
    def compute_metrics(eval_pred):
        load_accuracy = load_metric("accuracy")
        load_f1 = load_metric("f1")
        
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
        return {"accuracy": accuracy, "f1": f1}
    

    def tt_split(self, verbose=False):
        # Create binary sentiment with progress bar
        tqdm.pandas(desc="Creating sentiment labels", disable=not verbose)
        self.df['sentiment'] = self.df['rating'].progress_apply(lambda x: 1 if x >= 4 else 0)
        
        # Select features and target
        X = self.df['text']
        y = self.df['sentiment']
        
        # Create train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        # Convert to lists with progress bars
        if verbose:
            print("Converting split data to lists...")
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