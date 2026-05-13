import wandb
from wandb.integration.xgboost import WandbCallback
from sklearn.model_selection import train_test_split
from src.pipeline import get_model_pipeline

import os
from dotenv import load_dotenv

load_dotenv()


def train_model(X, y):


    # Initialize W&B run
    run = wandb.init(project="ecommerce-churn", job_type="train")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline = get_model_pipeline()
    
    # Train and log metrics to W&B
    pipeline.fit(X_train, y_train)
    
    # Log accuracy or other metrics
    score = pipeline.score(X_test, y_test)
    wandb.log({"test_accuracy": score})
    
    run.finish()
    return pipeline
