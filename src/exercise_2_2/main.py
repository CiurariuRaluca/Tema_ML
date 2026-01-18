import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

sys.path.insert(0, "..")

from exercise_2_1.Logistic_regression import MyLogisticRegression
from exercise_2_2.preprocess_dataset import build_sauce_dataset, SAUCES


def train_models_for_sauces(
    train_features, train_targets, sauces=SAUCES, lr=0.1, iters=25
):
    sauce_models = {}

    for sauce_name in sauces:
        y_train_binary = train_targets[sauce_name].values.astype(int)

        train_features_no_current = train_features.drop(
            columns=[sauce_name], errors="ignore"
        )
        feature_columns_used = train_features_no_current.columns

        clf = MyLogisticRegression(learning_rate=lr)
        clf.X = train_features_no_current.values
        clf.y = y_train_binary
        clf.newton_method(iterations=iters)

        sauce_models[sauce_name] = (clf, feature_columns_used)

    return sauce_models


def recommend_topk_for_receipt(
    receipt_id, features_no_sauce, sauce_models, sauces=SAUCES, K=3
):
    sauce_probabilities = {}

    sauces_already_in_cart = {
        s for s in sauces if features_no_sauce.loc[receipt_id, s] > 0
    }

    for sauce_name in sauces:
        if sauce_name in sauces_already_in_cart:
            continue

        clf, feature_columns_used = sauce_models[sauce_name]

        x_row = features_no_sauce.loc[[receipt_id], feature_columns_used].values
        prob = clf.foward(x_row)[0]

        sauce_probabilities[sauce_name] = float(prob)

    topk = sorted(sauce_probabilities.items(), key=lambda x: x[1], reverse=True)[:K]
    return [name for name, _ in topk]


def evaluate_recommender(sauce_models, test_features, test_targets, sauces=SAUCES, K=3):
    test_features_no_sauce = test_features.copy()
    test_features_no_sauce[sauces] = 0

    hit_scores = []
    precision_scores = []

    for receipt_id in test_features.index:
        true_sauces = {s for s in sauces if test_targets.loc[receipt_id, s] == 1}
        if len(true_sauces) == 0:
            continue

        recommendations = recommend_topk_for_receipt(
            receipt_id, test_features_no_sauce, sauce_models, sauces=sauces, K=K
        )
        rec_set = set(recommendations)

        hit_scores.append(int(len(true_sauces & rec_set) > 0))
        precision_scores.append(len(true_sauces & rec_set) / K)

    return float(np.mean(hit_scores)), float(np.mean(precision_scores))


def build_popularity_baseline(train_targets, sauces=SAUCES):
    sauce_counts = {s: int(train_targets[s].sum()) for s in sauces}
    sauces_sorted_by_popularity = sorted(
        sauce_counts.keys(), key=lambda s: sauce_counts[s], reverse=True
    )
    return sauces_sorted_by_popularity, sauce_counts


def recommend_baseline_popularity(receipt_id, sauces_sorted_by_popularity, K=3):
    return sauces_sorted_by_popularity[:K]


def evaluate_baseline_popularity(
    test_targets, sauces_sorted_by_popularity, sauces=SAUCES, K=3
):
    hit_scores = []
    precision_scores = []

    for receipt_id in test_targets.index:
        true_sauces = {s for s in sauces if test_targets.loc[receipt_id, s] == 1}
        if len(true_sauces) == 0:
            continue

        recommendations = recommend_baseline_popularity(
            receipt_id, sauces_sorted_by_popularity, K=K
        )
        rec_set = set(recommendations)

        hit_scores.append(int(len(true_sauces & rec_set) > 0))
        precision_scores.append(len(true_sauces & rec_set) / K)

    return float(np.mean(hit_scores)), float(np.mean(precision_scores))


if __name__ == "__main__":
    all_features, sauce_targets = build_sauce_dataset(
        "../ap_dataset.csv", sauces=SAUCES
    )

    X_train, X_test, Y_train, Y_test = train_test_split(
        all_features, sauce_targets, test_size=0.2, random_state=43
    )

    sauce_models = train_models_for_sauces(
        X_train, Y_train, sauces=SAUCES, lr=0.1, iters=25
    )
    print("trained models:", list(sauce_models.keys()))

    for K in [1, 3, 5]:
        hit_k, precision_k = evaluate_recommender(
            sauce_models, X_test, Y_test, sauces=SAUCES, K=K
        )
        print(f"models:   K={K}  Hit@K={hit_k:.3f}  Precision@K={precision_k:.3f}")

    sauces_sorted_by_popularity, sauce_counts = build_popularity_baseline(
        Y_train, sauces=SAUCES
    )
    print("\npopular order:", sauces_sorted_by_popularity)
    print("counts:", sauce_counts)

    for K in [1, 3, 5]:
        hit_k, precision_k = evaluate_baseline_popularity(
            Y_test, sauces_sorted_by_popularity, sauces=SAUCES, K=K
        )
        print(f"baseline: K={K}  Hit@K={hit_k:.3f}  Precision@K={precision_k:.3f}")
