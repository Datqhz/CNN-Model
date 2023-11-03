import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


def training_model():
    df = pd.read_csv('emotion_text.csv')

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])
    X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, df['label'], test_size=0.2, random_state=42)
    clf = svm.SVC(kernel='linear')

    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)

    y_preds = clf.predict(X_test)

    print("Classifier metrics on the test set")
    print(f"Accurracy: {accuracy_score(y_test, y_preds)*100:.2f}%")
    print(f"Precision: {precision_score(y_test, y_preds, average='macro')}")
    print(f"Recall: {recall_score(y_test, y_preds, average='macro')}")
    print(f"F1: {f1_score(y_test, y_preds, average='macro')}")

    # Save an extisting model to file
    pickle.dump(clf, open("linearSVC_model.pkl", "wb"))
    pickle.dump(tfidf_vectorizer, open("tfidf_vectorize.pkl", "wb"))