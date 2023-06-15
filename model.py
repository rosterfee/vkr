import numpy as np
import pickle
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_model_score(X_train, X_test, y_train, y_test):
    # clf = MLPClassifier(random_state=42, max_iter=3000)
    # clf = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=5)
    # clf = DecisionTreeClassifier(random_state=42)
    # clf = ComplementNB()
    # clf = SVC(random_state=42, kernel='poly', class_weight='balanced', probability=True)
    # clf = KNeighborsClassifier(leaf_size=1, n_neighbors=13, weights='distance')
    # clf = LogisticRegression(fit_intercept=False, n_jobs=-1, warm_start=True)
    clf = MultinomialNB(alpha=0.001, force_alpha=True, fit_prior=True)
    pipeline = Pipeline([('scaler', MinMaxScaler()), ('nb', clf)])

    loo = LeaveOneOut()

    scores = []
    for train_index, test_index in loo.split(X_train, y_train):
        fold_X_train, fold_X_test = X_train.iloc[train_index], X_train.iloc[test_index]
        fold_y_train, fold_y_test = y_train[train_index], y_train[test_index]

        pipeline.fit(fold_X_train, fold_y_train)

        y_pred = pipeline.predict(fold_X_test)
        score = accuracy_score(fold_y_test, y_pred)

        scores.append(score)

    print(f'Средний скор модели на кросс-валидации: {round(float(np.mean(scores)), 2)}')

    # Сохранение обученной модели
    filename = "model.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(pipeline, file)

    y_pred = pipeline.predict_proba(X_test)[:, 1]
    return round(roc_auc_score(y_test, y_pred), 2)


def predict_class(sample):
    pipeline = pickle.load(open('model.pkl', 'rb'))
    return pipeline.predict(sample)[0]
