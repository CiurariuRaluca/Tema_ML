import pandas as pd
import numpy as np


def get_dataset(path_csv="../ap_dataset.csv"):
    dataset = pd.read_csv(path_csv, encoding="utf-8-sig")

    # data cleaning: verificam daca nu exista valori lipsa si nu exista preturi gresite(<=0),
    # si standardizam denumirile produselor prin eliminarea spatiilor pentru a evita inconsecvente
    # la construirea targetului si a vectorului de produse
    dataset["retail_product_name"] = (
        dataset["retail_product_name"].astype(str).str.strip()
    )

    dataset["data_bon"] = pd.to_datetime(
        dataset["data_bon"]
    )  # converteste din string in datetime object

    # tipuri de features:

    objects = dataset.dtypes == "object"  # features categoriale
    object_cols = list(objects[objects].index)

    int_ = dataset.dtypes == "int"  # features cu valori int
    int_cols = list(int_[int_].index)

    float_ = dataset.dtypes == "float"  # features cu valori float
    float_cols = list(float_[float_].index)

    # feature engineering - creare features temporale(ora, zi...)
    # poate ora/ziua, weekend influenteaza cumpararea sosului
    dataset["hour"] = dataset["data_bon"].dt.hour
    dataset["day_of_week"] = (
        dataset["data_bon"].dt.dayofweek + 1
    )  # .dt.dayofweek -> return 0-6
    dataset["is_weekend"] = dataset["day_of_week"].isin([6, 7]).astype(int)
    # de ce facem feature engineering:
    # -features categoriale -> one hot ecoding
    # -numerice(float/int) -> normalizare/standardizare
    # deoarece aceste features temporale pot fi predictori buni

    # filtram bonurile - le selectam pe cele care contin Crazy Schnitzel
    filtered_dataset = dataset[dataset["retail_product_name"] == "Crazy Schnitzel"][
        "id_bon"
    ].unique()
    schnitzel_dataset = dataset[dataset["id_bon"].isin(filtered_dataset)].copy()

    # creare target
    target = schnitzel_dataset.groupby("id_bon")["retail_product_name"].apply(
        lambda products: int("Crazy Sauce" in products.values)
    )

    schnitzel_no_sauce = schnitzel_dataset[
        schnitzel_dataset["retail_product_name"] != "Crazy Sauce"
    ].copy()

    # features agregate per bon (fara Crazy Sauce)
    cart_features = (
        schnitzel_no_sauce.groupby("id_bon")
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

    cart_features.columns = [
        "id_bon",
        "total_value",        
        "avg_price",         
        "cart_size",          
        "hour",
        "day_of_week",
        "is_weekend",
        "distinct_products",  
    ]

    # construire vector de produse
    X_products = pd.crosstab(
        schnitzel_dataset["id_bon"], schnitzel_dataset["retail_product_name"]
    )

    # scoatem Crazy Sauce si Crazy Schnitzel din features ca sa evitam leakage
    for col in ["Crazy Sauce", "Crazy Schnitzel"]:
        if col in X_products.columns:
            X_products.drop(col, axis=1, inplace=True)

    X_numerical = cart_features.set_index("id_bon")[
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

    X = X_numerical.join(X_products, how="left").fillna(0)
    y = target.loc[X.index].astype(int)

    return X, y


def get_eda_objects(path_csv="../ap_dataset.csv"):
    dataset = pd.read_csv(path_csv, encoding="utf-8-sig")

    dataset["retail_product_name"] = (
        dataset["retail_product_name"].astype(str).str.strip()
    )
    dataset["data_bon"] = pd.to_datetime(dataset["data_bon"])

    # feature engineering temporal
    dataset["hour"] = dataset["data_bon"].dt.hour
    dataset["day_of_week"] = dataset["data_bon"].dt.dayofweek + 1
    dataset["is_weekend"] = dataset["day_of_week"].isin([6, 7]).astype(int)

    # filtram bonurile care contin Crazy Schnitzel
    filtered_dataset = dataset[dataset["retail_product_name"] == "Crazy Schnitzel"][
        "id_bon"
    ].unique()
    schnitzel_dataset = dataset[dataset["id_bon"].isin(filtered_dataset)].copy()

    # target: Crazy Sauce exista in bon
    target = schnitzel_dataset.groupby("id_bon")["retail_product_name"].apply(
        lambda products: int("Crazy Sauce" in products.values)
    )

    schnitzel_no_sauce = schnitzel_dataset[
        schnitzel_dataset["retail_product_name"] != "Crazy Sauce"
    ].copy()

    # features agregate per bon (fara Crazy Sauce)
    cart_features = (
        schnitzel_no_sauce.groupby("id_bon")
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

    cart_features.columns = [
        "id_bon",
        "total_value",        
        "avg_price",          
        "cart_size",          
        "hour",
        "day_of_week",
        "is_weekend",
        "distinct_products", 
    ]

    numerical_cols = [
        "hour",
        "day_of_week",
        "is_weekend",
        "total_value",
        "avg_price",
        "cart_size",
        "distinct_products",
    ]

    cart_features_with_target = cart_features.set_index("id_bon").copy()
    cart_features_with_target["target"] = target

    return (
        cart_features_with_target,
        numerical_cols,
        schnitzel_dataset,
        target,
        cart_features,
    )


def standardize_prepocessed_data(X_train, X_test):
    num_cols = [
        "hour",
        "day_of_week",
        "is_weekend",
        "total_value",
        "avg_price",
        "cart_size",
        "distinct_products",
    ]

    X_train_s = X_train.copy()
    X_test_s = X_test.copy()

    num_cols = [c for c in num_cols if c in X_train_s.columns]

    mu = X_train_s[num_cols].mean(axis=0)
    sigma = X_train_s[num_cols].std(axis=0, ddof=0)

    sigma = sigma.replace(0, 1)

    X_train_s[num_cols] = (X_train_s[num_cols] - mu) / sigma
    X_test_s[num_cols] = (X_test_s[num_cols] - mu) / sigma

    return X_train_s, X_test_s


def feature_selection(X_train, X_test, selected_num_cols):
    X_train_sp = X_train.copy()
    X_test_sp = X_test.copy()

    mu = X_train_sp[selected_num_cols].mean(axis=0)
    sigma = X_train_sp[selected_num_cols].std(axis=0, ddof=0)

    sigma = sigma.replace(0, 1)

    # standardizare doar pentru coloanele selectate
    X_train_sp[selected_num_cols] = (X_train_sp[selected_num_cols] - mu) / sigma
    X_test_sp[selected_num_cols] = (X_test_sp[selected_num_cols] - mu) / sigma

    # eliminare coloane constante (std = 0)
    stds = X_train_sp.std(axis=0)
    const_cols = stds[stds == 0].index

    X_train_sp = X_train_sp.drop(columns=const_cols)
    X_test_sp = X_test_sp.drop(columns=const_cols)

    # eliminare coloane duplicate (identice)
    dup_mask = X_train_sp.T.duplicated()

    X_train_sp = X_train_sp.loc[:, ~dup_mask]
    X_test_sp = X_test_sp[X_train_sp.columns]

    return X_train_sp, X_test_sp
