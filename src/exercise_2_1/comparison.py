from sklearn.linear_model import LogisticRegression as LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from preprocess_dataset import get_dataset

X, y = get_dataset()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=43, stratify=y
)


clf = LogisticRegression(
    solver="newton-cg",# asemantor cu newton s method
    penalty=None,
    max_iter=15
)
clf1 = LogisticRegression(# asemnator cu gd
    solver="sag",
    penalty=None,
    max_iter=900
)

clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
print("train accuracy (clf):", accuracy_score(y_train, y_pred_train))

y_pred_test = clf.predict(X_test)
print("test accuracy (clf):", accuracy_score(y_test, y_pred_test))

clf1.fit(X_train, y_train)

y_pred_train = clf1.predict(X_train)
print("train accuracy (clf1):", accuracy_score(y_train, y_pred_train))

y_pred_test = clf1.predict(X_test)
print("test accuracy (clf1):", accuracy_score(y_test, y_pred_test))