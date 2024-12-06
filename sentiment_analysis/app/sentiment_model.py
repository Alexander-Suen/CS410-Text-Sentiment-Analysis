import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, pipeline
import torch
from torch.nn.functional import softmax
import numpy as np
from datasets import load_dataset, Dataset  # Removed load_metric
import evaluate  # Added this import
from pathlib import Path


class SentimentAnalysisModel:
    def __init__(self, verbose=False, load_pretrained=True):
        self.model_dir = Path("models/sentiment_model")
        print(f"Looking for model in: {self.model_dir.absolute()}")  # Debug print
        self.is_trained = False
        
        if load_pretrained and self.model_dir.exists():
            print(f"Found model directory at {self.model_dir}") if verbose else None
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
                self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
                self.is_trained = True
                print("Successfully loaded pre-trained model and tokenizer") if verbose else None
                return  # Exit initialization here if model is loaded successfully
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                print("Falling back to default model")
        else:
            print(f"No saved model found at {self.model_dir}, using default model") if verbose else None
        
        # Only reach here if no saved model was found or there was an error loading it
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        
        # Training setup only needed if we don't have a pre-trained model
        print("Loading dataset...") if verbose else None
        self.dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty")
        
        print("Converting to DataFrame...") if verbose else None
        self.df = pd.DataFrame(self.dataset['full'])
        print(self.df.columns)
        
        print("Splitting data...") if verbose else None
        self.train_data, self.test_data = self.tt_split(verbose)
        
        self.tokenized_train = Dataset.from_dict(self.train_data).map(self.preprocess_function, batched=True)
        self.tokenized_test = Dataset.from_dict(self.train_data).map(self.preprocess_function, batched=True)
        

    def preprocess_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True)
    
   
    def compute_metrics(self, eval_pred):
        # Update the metrics computation
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = f1_metric.compute(predictions=predictions, references=labels, average='binary')["f1"]
        return {"accuracy": accuracy, "f1": f1}
    

    def tt_split(self, verbose=False):
        # Create binary sentiment with progress bar
        tqdm.pandas(desc="Creating sentiment labels", disable=not verbose)
        self.df['sentiment'] = self.df['rating'].progress_apply(lambda x: 1 if x >= 4 else 0)

        # Select features and target
        X = self.df['text']
        y = self.df['sentiment']

        # First, take only 11000 points (10000 for train + 1000 for test)
        if len(X) > 11000:
            if verbose:
                print(f"Reducing dataset from {len(X)} to 11000 points...")
            indices = np.random.choice(len(X), 11000, replace=False)
            X = X.iloc[indices]
            y = y.iloc[indices]

        # Create train/test split
        X_train, X_test, y_train, y_test = evaluate.train_test_split(
            X, y,
            test_size=1000,  # Exact number for test set
            random_state=42,
            stratify=y
        )

        # Take only first 10000 points for training if more exist
        if len(X_train) > 10000:
            X_train = X_train[:10000]
            y_train = y_train[:10000]

        # Convert to lists with progress bars
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

    
    def train_model(self):
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        training_args = TrainingArguments(
            output_dir="./checkpoints",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            save_strategy="epoch",        # Save at each epoch
            evaluation_strategy="epoch",  # Added this line to match save_strategy
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy"  # Added to specify what metric to use for "best" model
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_train,
            eval_dataset=self.tokenized_test,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        trainer.train()
        
        # Save the final model
        print("Saving trained model...")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(self.model_dir))
        self.tokenizer.save_pretrained(str(self.model_dir))
        self.is_trained = True
        
    def eval_model(self):
        self.trainer.evaluate()

    
    def analyze_text(self, text):
        self.model.eval()

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        probabilities = softmax(logits, dim=1)

        positive_score = probabilities[0][1].item()

        return positive_score