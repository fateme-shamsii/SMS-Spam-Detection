"""
Model training and evaluation for SMS Spam Detection (Phase 03).
Contains vectorizers, baseline models, and train/evaluate logic for Code Organization.
"""
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

RANDOM_STATE = 42
DEFAULT_MAX_FEATURES = 10000
DEFAULT_MIN_DF = 2


def get_baseline_models(random_state=RANDOM_STATE):
    """
    Return baseline models required in Phase 03.
    """
    return {
        "Naive Bayes (MultinomialNB)": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "SVM (LinearSVC)": LinearSVC(max_iter=2000, random_state=random_state),
    }


def get_vectorizers(max_features=DEFAULT_MAX_FEATURES, min_df=DEFAULT_MIN_DF):
    """
    Return dict of (name -> fitted vectorizer is not fitted here; we fit on X_train).
    Returns vectorizer instances: BoW, TF-IDF, TF-IDF n-gram.
    """
    return {
        "Bag of Words": CountVectorizer(
            max_features=max_features, min_df=min_df, lowercase=True
        ),
        "TF-IDF": TfidfVectorizer(
            max_features=max_features, min_df=min_df, lowercase=True
        ),
        "TF-IDF (unigram+bigram)": TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            lowercase=True,
            ngram_range=(1, 2),
        ),
    }


def train_baseline_models(X_train, y_train, X_val, y_val, random_state=RANDOM_STATE):
    """
    Train the three baseline classifiers (MultinomialNB, LogisticRegression, LinearSVC)
    on (X_train, y_train) and evaluate on (X_val, y_val).
    Returns (trained_models_dict, results_list of dicts with Model, Accuracy, Precision, Recall, F1-Score).
    """
    models = get_baseline_models(random_state=random_state)
    trained = {}
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        y_pred = model.predict(X_val)
        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_val, y_pred),
                "Precision": precision_score(y_val, y_pred, zero_division=0),
                "Recall": recall_score(y_val, y_pred, zero_division=0),
                "F1-Score": f1_score(y_val, y_pred, zero_division=0),
            }
        )
    return trained, results


def run_feature_model_grid(
    X_train,
    y_train,
    X_val,
    y_val,
    vectorizers=None,
    models=None,
    random_state=RANDOM_STATE,
):
    """
    Evaluate ALL (vectorizer × model) combinations on the validation set.

    Why: Different models prefer different feature spaces (e.g. NB often works well with BoW counts,
    SVM often works well with TF-IDF). This function implements the strict-professor requested logic.

    Returns:
        results: list of dicts with Feature, Model, Accuracy, Precision, Recall, F1-Score, ROC-AUC
    """
    if vectorizers is None:
        vectorizers = get_vectorizers()
    if models is None:
        models = get_baseline_models(random_state=random_state)

    results = []
    for feat_name, vec in vectorizers.items():
        X_tr = vec.fit_transform(X_train)
        X_va = vec.transform(X_val)

        for model_name, model in models.items():
            model.fit(X_tr, y_train)
            y_pred = model.predict(X_va)
            
            # Calculate ROC-AUC
            try:
                # For models with decision_function (SVM, LogisticRegression)
                y_score = model.decision_function(X_va)
                roc_auc = roc_auc_score(y_val, y_score)
            except AttributeError:
                # For MultinomialNB with predict_proba
                try:
                    y_proba = model.predict_proba(X_va)[:, 1]
                    roc_auc = roc_auc_score(y_val, y_proba)
                except:
                    roc_auc = 0.0
            
            results.append(
                {
                    "Feature": feat_name,
                    "Model": model_name,
                    "Accuracy": accuracy_score(y_val, y_pred),
                    "Precision": precision_score(y_val, y_pred, zero_division=0),
                    "Recall": recall_score(y_val, y_pred, zero_division=0),
                    "F1-Score": f1_score(y_val, y_pred, zero_division=0),
                    "ROC-AUC": roc_auc,
                }
            )
    return results