import time
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score


def make_grid():
    return {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }


def fit_transform_tdidf(X_train, X_test):
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf


def perform_grid_search(X_train, y_train, param_grid):
    rf_classifier = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf_classifier,
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def evaluate_model(model, X_test, y_test):
    start_time = time.time()
    y_pred = model.predict(X_test)
    print("Inference time: ", round(time.time() - start_time, 2))
    report = classification_report(y_test, y_pred)
    print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))
    print("F1 Score on Test Set:", f1_score(y_test, y_pred))
    print("ROC-AUC on Test Set:", roc_auc_score(y_test, y_pred))
    print("Classification Report:\n", report)
    return
