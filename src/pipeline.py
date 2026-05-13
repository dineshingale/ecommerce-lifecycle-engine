from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from wandb.integration.xgboost import WandbCallback

def get_model_pipeline():
    pipeline = Pipeline([

        # fit Imputation and scaling only on train data — to prevents leakage

        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            n_estimators=100,
            callbacks=[WandbCallback()]
        ))
    ])
    
    return pipeline
