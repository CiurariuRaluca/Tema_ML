import pandas as pd
from Logistic_regression import MyLogisticRegression
from sklearn.model_selection import train_test_split
from evaluation_model import evaluate_model, majority_baseline
from save_coeficients import save_model_coefficients
from preprocess_dataset import get_dataset, standardize_prepocessed_data
import numpy as np

X, y = get_dataset()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=43, stratify=y
)

model = MyLogisticRegression(learning_rate=0.1)
model.X = X_train.values
model.y = y_train.values
model.gradient_descent(iterations=2000)

model1 = MyLogisticRegression(learning_rate=0.1)
model1.X = X_train.values
model1.y = y_train.values
model1.newton_method(iterations=15)

y_pred_train = model1.predict(X_train)
accuracy = np.mean(y_pred_train == y_train.values)
print(accuracy)

y_predicted = model1.predict(X_test)
accuracy = np.mean(y_predicted == y_test.values)
print(accuracy)

y_pred_train1 = model.predict(X_train)
accuracy = np.mean(y_pred_train1 == y_train.values)
print(accuracy)

y_predicted1 = model.predict(X_test)
accuracy = np.mean(y_predicted1 == y_test.values)
print(accuracy)

##acuratete pe date standardizate
X_train_s, X_test_s = standardize_prepocessed_data(X_train, X_test)

model3 = MyLogisticRegression(learning_rate=0.1)
model3.X = X_train_s.values
model3.y = y_train.values
model3.gradient_descent(iterations=800)

model2 = MyLogisticRegression(learning_rate=0.1)
model2.X = X_train_s.values
model2.y = y_train.values
model2.newton_method(iterations=15)

print("Standardized data:acuracy\n")

y_pred_train = model3.predict(X_train_s)
accuracy = np.mean(y_pred_train == y_train.values)
print(accuracy)

y_predicted = model3.predict(X_test_s)
accuracy = np.mean(y_predicted == y_test.values)
print(accuracy)

y_pred_train1 = model2.predict(X_train_s)
accuracy = np.mean(y_pred_train1 == y_train.values)
print(accuracy)

y_predicted1 = model2.predict(X_test_s)
accuracy = np.mean(y_predicted1 == y_test.values)
print(accuracy)

models_to_save = {
    "GD_2_iter500": model,
    "Newton.1_iter15": model1,
    "GD_std1_iter800_st": model3,
    "Newton_std_1_iter15_st": model2,
   
}
feature_names = X_train_s.columns

for name, m in models_to_save.items():
    save_model_coefficients(m, feature_names, model_name=name, out_dir="../../results")

all_results = []

all_results.append(majority_baseline(y_train, y_test))

res_gd, _, _ = evaluate_model(model, X_train, y_train, X_test, y_test, model_name="GD RAW")
all_results.append(res_gd)

res_newton, _, _ = evaluate_model(model1, X_train, y_train, X_test, y_test, model_name="Newton RAW")
all_results.append(res_newton)

res_gd_std, _, _ = evaluate_model(model3, X_train_s, y_train, X_test_s, y_test, model_name="GD STD")
all_results.append(res_gd_std)

res_newton_std, _, _ = evaluate_model(model2, X_train_s, y_train, X_test_s, y_test, model_name="Newton STD")
all_results.append(res_newton_std)

pd.DataFrame(all_results).to_csv("../../result_metrics/results_metrics.csv", index=False, encoding="utf-8-sig")



