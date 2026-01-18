import pandas as pd
from Logistic_regression import MyLogisticRegression
from sklearn.model_selection import train_test_split
from evaluation_model import evaluate_model, majority_baseline
from save_coeficients import save_model_coefficients
from preprocess_dataset import get_dataset, feature_selection
import numpy as np

X, y = get_dataset()

selected_num_cols = ["hour", "day_of_week", "is_weekend", "cart_size"]

all_num_cols = [
    "hour",
    "day_of_week",
    "is_weekend",
    "total_value",
    "avg_price",
    "cart_size",
    "distinct_products",
]
product_cols = [c for c in X.columns if c not in all_num_cols]

X = X[selected_num_cols + product_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=43, stratify=y
)

X_train_sp, X_test_sp = feature_selection(X_train, X_test, selected_num_cols)

model = MyLogisticRegression(learning_rate=0.2)
model.X = X_train_sp.values
model.y = y_train.values
model.gradient_descent(iterations=900)

model1 = MyLogisticRegression(learning_rate=0.1)
model1.X = X_train_sp.values
model1.y = y_train.values
model1.newton_method(iterations=10)

y_pred_train = model.predict(X_train_sp.values)
print("Train acc GD:", np.mean(y_pred_train == y_train.values))

y_pred_test = model.predict(X_test_sp.values)
print("Test acc GD:", np.mean(y_pred_test == y_test.values))

y_pred_train2 = model1.predict(X_train_sp.values)
print("Train acc Newton:", np.mean(y_pred_train2 == y_train.values))

y_pred_test2 = model1.predict(X_test_sp.values)
print("Test acc Newton:", np.mean(y_pred_test2 == y_test.values))

models_to_save = {
    "ft_GD_lr1_iter900": model,
    "ft_Newton_lr1_iter15": model1,
}

feature_names = X_train_sp.columns

for name, m in models_to_save.items():
    save_model_coefficients(m, feature_names, model_name=name, out_dir="../../results")

all_results = []

all_results.append(majority_baseline(y_train, y_test))

res_gd, _, _ = evaluate_model(
    model, X_train_sp, y_train, X_test_sp, y_test, model_name="GD FS+STD"
)
all_results.append(res_gd)

res_newton, _, _ = evaluate_model(
    model1, X_train_sp, y_train, X_test_sp, y_test, model_name="Newton FS+STD"
)
all_results.append(res_newton)

pd.DataFrame(all_results).to_csv(
    "../../result_metrics/results_metrics_ft.csv", index=False, encoding="utf-8-sig"
)
