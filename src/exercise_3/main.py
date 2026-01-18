import numpy as np
import pandas as pd
import random

data = pd.read_csv("../ap_dataset.csv", encoding="utf-8-sig")
data["data_bon"] = pd.to_datetime(data["data_bon"])
data["retail_product_name"] = data["retail_product_name"].astype(str).str.strip()

data["hour"] = data["data_bon"].dt.hour
data["day_of_week"] = data["data_bon"].dt.dayofweek + 1
data["is_weekend"] = data["day_of_week"].isin([6, 7]).astype(int)

carts = (
    data.groupby("id_bon")
    .agg(
        total_value=("SalePriceWithVAT", "sum"),
        cart_size=("SalePriceWithVAT", "count"),
        distinct_products=("retail_product_name", "nunique"),
        hour=("hour", "first"),
        day_of_week=("day_of_week", "first"),
        is_weekend=("is_weekend", "first"),
        data_bon=("data_bon", "first"),
    )
    .reset_index()
)

product_matrix = pd.crosstab(data["id_bon"], data["retail_product_name"])

numeric_features = carts.set_index("id_bon")[
    [
        "hour",
        "day_of_week",
        "is_weekend",
        "cart_size",
        "distinct_products",
        "total_value",
    ]
]
features = numeric_features.join(product_matrix, how="left").fillna(0)

avg_price_per_product = (
    data.groupby("retail_product_name")["SalePriceWithVAT"].mean().to_dict()
)


def get_price(product_name):
    return float(avg_price_per_product.get(product_name, 0.0))


SAUCES = [
    "Crazy Sauce",
    "Cheddar Sauce",
    "Extra Cheddar Sauce",
    "Garlic Sauce",
    "Tomato Sauce",
    "Blueberry Sauce",
    "Spicy Sauce",
    "Pink Sauce",
]
candidate_products = [p for p in SAUCES if p in features.columns]

sorted_carts = carts.sort_values("data_bon")
split_index = int(0.8 * len(sorted_carts))

train_receipts = sorted_carts.iloc[:split_index]["id_bon"].values
test_receipts = sorted_carts.iloc[split_index:]["id_bon"].values

X_train = features.loc[train_receipts]
X_test = features.loc[test_receipts]


class MyNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        m, _ = X.shape

        self.p_y1 = (y.sum() + self.alpha) / (m + 2 * self.alpha)
        self.p_y0 = 1 - self.p_y1

        X1 = X[y == 1]
        X0 = X[y == 0]

        self.theta1 = (X1.sum(axis=0) + self.alpha) / (len(X1) + 2 * self.alpha)
        self.theta0 = (X0.sum(axis=0) + self.alpha) / (len(X0) + 2 * self.alpha)

        self.log_p_y1 = np.log(self.p_y1)
        self.log_p_y0 = np.log(self.p_y0)

        self.log_theta1 = np.log(self.theta1)
        self.log_1m_theta1 = np.log(1 - self.theta1)

        self.log_theta0 = np.log(self.theta0)
        self.log_1m_theta0 = np.log(1 - self.theta0)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)

        log1 = self.log_p_y1 + (X @ self.log_theta1) + ((1 - X) @ self.log_1m_theta1)
        log0 = self.log_p_y0 + (X @ self.log_theta0) + ((1 - X) @ self.log_1m_theta0)

        mx = np.maximum(log0, log1)
        p1 = np.exp(log1 - mx) / (np.exp(log0 - mx) + np.exp(log1 - mx))
        return p1


binary_train = (X_train > 0).astype(int)
nb_models = {}

for product in candidate_products:
    y_train = (X_train[product] > 0).astype(int).values
    X_train_no_product = binary_train.drop(columns=[product], errors="ignore").values

    model = MyNaiveBayes(alpha=1.0)
    model.fit(X_train_no_product, y_train)
    nb_models[product] = model


def rank_products_for_cart(cart_vector):
    cart_binary = (cart_vector > 0).astype(int)
    scored = []

    for product in candidate_products:
        if cart_vector.get(product, 0) > 0:
            continue

        x_input = cart_binary.drop(labels=[product], errors="ignore").values.reshape(
            1, -1
        )
        prob = float(nb_models[product].predict_proba(x_input)[0])
        score = prob * get_price(product)

        scored.append((product, score))

    scored.sort(key=lambda t: t[1], reverse=True)
    return [p for p, _ in scored]


def evaluate_hit_at_k(K_values=(1, 3, 5), seed=42):
    random.seed(seed)
    hits = {K: 0 for K in K_values}
    total = 0

    valid_test_receipts = X_test[(X_test[candidate_products].sum(axis=1) > 0)].index

    for receipt_id in valid_test_receipts:
        full_cart = X_test.loc[receipt_id].copy()
        present_products = [p for p in candidate_products if full_cart[p] > 0]

        removed_product = random.choice(present_products)
        full_cart[removed_product] = 0

        ranked_list = rank_products_for_cart(full_cart)

        total += 1
        for K in K_values:
            if removed_product in ranked_list[:K]:
                hits[K] += 1

    return total, {K: hits[K] / total for K in K_values}


total_cases, hit_nb = evaluate_hit_at_k()
print(f"\n naive nayes ranking | test_cases={total_cases}")
print("hit@1/3/5:", hit_nb)


popularity_order = (
    (X_train[candidate_products] > 0).mean(axis=0).sort_values(ascending=False).index
).tolist()


def evaluate_popularity_baseline(K_values=(1, 3, 5), seed=42):
    random.seed(seed)
    hits = {K: 0 for K in K_values}
    total = 0

    valid_test_receipts = X_test[(X_test[candidate_products].sum(axis=1) > 0)].index

    for receipt_id in valid_test_receipts:
        real_cart = X_test.loc[receipt_id]
        present_products = [p for p in candidate_products if real_cart[p] > 0]
        removed_product = random.choice(present_products)

        total += 1
        for K in K_values:
            if removed_product in popularity_order[:K]:
                hits[K] += 1

    return total, {K: hits[K] / total for K in K_values}


_, hit_pop = evaluate_popularity_baseline()
print("\npopularity baseline")
print("hit@1/3/5:", hit_pop)
