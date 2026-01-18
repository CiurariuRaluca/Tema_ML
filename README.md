# Tema ML
---

## Cerințe implementate

### 2.1 Logistic Regression — “Will the client include the sauce?”
**Task:** doar bonurile care conțin `Crazy Schnitzel`  
**Target:** `y = 1` dacă bonul conține și `Crazy Sauce`, altfel `0`

**Features:**
- vector produse per bon (count per produs)
- agregări pe bon: `cart_size`, `distinct_products`, `total_value`, `avg_price`
- context temporal: `hour`, `day_of_week`, `is_weekend`
- anti-leakage: `Crazy Sauce` este exclus din features

**Regresie Logistică implementată manual:**
- Gradient Descent
- Newton Method

Evaluare:
- accuracy, precision/recall/F1, ROC-AUC
- confusion matrix
- baseline majority class  
Rezultatele sunt salvate în CSV.

---

### 2.2 Logistic Regression — câte un model pentru fiecare sos + recomandare
Se antrenează câte un model pentru fiecare sos:

- Crazy Sauce  
- Cheddar Sauce  
- Extra Cheddar Sauce  
- Garlic Sauce  
- Tomato Sauce  
- Blueberry Sauce  
- Spicy Sauce  
- Pink Sauce  

Pentru sosul `s`, targetul este:
- `y_s = 1` dacă sosul apare în bon, altfel `0`

**Anti-leakage:** sosul curent este eliminat din features la antrenare.

**Pseudo-recomandare Top-K:**  
Pentru un coș fără sosuri, se calculează `P(s | coș)` pentru fiecare sos și se recomandă Top-K.

Evaluare:
- Hit@K și Precision@K (K=1,3,5)
- baseline popularitate globală (Top-K cele mai frecvente sosuri)

---

### 3 Ranking 
Se construiește un ranking de produse candidate pentru upsell (ex: sosuri) folosind:

- **Naive Bayes**
- scor: `Score(p | coș) = P(p | coș) * price(p)`  
  (maximizarea valorii așteptate)

Cadru experimental:
- pentru fiecare bon din test se elimină 1 produs și se verifică dacă apare în Top-K ranking

Evaluare:
- Hit@1 / Hit@3 / Hit@5
- baseline popularitate globală

---
Structura:
TEMA_ML/
│── .venv/
│── plots/
│── result_metrics/
│ ├── results_metrics.csv
│ ├── results_metrics_ft.csv
│ └── results_metrics_l2.csv
│── results/
│ ├── coefficients_*.csv
│── src/
│ ├── exercise_2_1/
│ ├── exercise_2_2/
│ └── exercise_3/
│── main.py
│── ap_dataset.csv
│── README.md
│── requirements.txt

##  Setup

### 1) Creează și activează venv

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

 Rulare cod
 
2.1 EDA (grafice)
python src/exercise_2_1/eda.py
Output:
grafice salvate în plots/

2.1 Logistic Regression + evaluare + coeficienți
python src/exercise_2_1/train.py

Output:
metrici: result_metrics/results_metrics.csv
coeficienți: results/coefficients_*.csv

2.2 Modele pentru fiecare sos + recomandări Top-K
python src/exercise_2_2/main.py

Output (în consolă):
Hit@K și Precision@K pentru recomandările bazate pe modele
Hit@K și Precision@K pentru baseline popularitate

3 Ranking (Naive Bayes) + Scor = P * price
python src/exercise_3/ranking_nb.py

Output:
Hit@1 / Hit@3 / Hit@5 pentru ranking
Hit@K baseline popularitate

Output-uri salvate
Metrici (CSV)

result_metrics/results_metrics.csv
result_metrics/results_metrics_ft.csv
result_metrics/results_metrics_l2.csv
Coeficienți modele (CSV)
results/coefficients_*.csv
Grafice EDA
plots/*.pdf sau plots/*.png