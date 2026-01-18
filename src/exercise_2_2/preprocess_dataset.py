import pandas as pd

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


def build_sauce_dataset(path_csv="../ap_dataset.csv", sauces=SAUCES):
    sales_data = pd.read_csv(path_csv, encoding="utf-8-sig")

    sales_data["retail_product_name"] = (
        sales_data["retail_product_name"].astype(str).str.strip()
    )
    sales_data["data_bon"] = pd.to_datetime(sales_data["data_bon"])

    sales_data["hour"] = sales_data["data_bon"].dt.hour
    sales_data["day_of_week"] = sales_data["data_bon"].dt.dayofweek + 1
    sales_data["is_weekend"] = sales_data["day_of_week"].isin([6, 7]).astype(int)

    cart_product_counts = pd.crosstab(
        sales_data["id_bon"], sales_data["retail_product_name"]
    )

    for sauce_name in sauces:
        if sauce_name not in cart_product_counts.columns:
            cart_product_counts[sauce_name] = 0

    sauce_targets = (cart_product_counts[sauces] > 0).astype(int)

    sales_no_sauces = sales_data[~sales_data["retail_product_name"].isin(sauces)].copy()

    cart_aggregates = (
        sales_no_sauces.groupby("id_bon")
        .agg(
            {
                "SalePriceWithVAT": ["sum", "mean", "count"],
                "hour": "first",
                "day_of_week": "first",
                "is_weekend": "first",
                "retail_product_name": "nunique",
            }
        )
        .reset_index()
    )

    cart_aggregates.columns = [
        "id_bon",
        "total_value",
        "avg_price",
        "cart_size",
        "hour",
        "day_of_week",
        "is_weekend",
        "distinct_products",
    ]

    cart_numeric_features = cart_aggregates.set_index("id_bon")[
        [
            "hour",
            "day_of_week",
            "is_weekend",
            "total_value",
            "avg_price",
            "cart_size",
            "distinct_products",
        ]
    ]

    X_features = cart_numeric_features.join(cart_product_counts, how="left").fillna(0)

    sauce_targets = sauce_targets.loc[X_features.index]

    return X_features, sauce_targets
