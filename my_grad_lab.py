# %%
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# %%
college_completion = pd.read_csv("cc_institution_details(2).csv")
college_completion.info()

# %%
# drop unnecessry columns
college_completion[college_completion['hbcu'] == 'X'][['pell_percentile', 'grad_100_percentile','long_x','lat_y','med_sat_percentile']]
college_completion_new = college_completion[['hbcu','pell_percentile', 'grad_100_percentile','long_x','lat_y','med_sat_percentile']]

college_completion_new.head()

# %%
# one hot encode target variable
college_completion_encoded = pd.get_dummies(college_completion_new, columns=['hbcu'])
college_completion_encoded["hbcu_X"] = (college_completion_encoded["hbcu_X"].astype(int))

college_completion_encoded.head()

# %%
# drop na values
college_completion_encoded = college_completion_encoded.dropna()

college_completion_encoded.head(20)

# %%
# calculate hbcu prevalence
hbcu_prevalence = college_completion_encoded["hbcu_X"].mean() * 100

print(str(hbcu_prevalence)[:4] + '%')

# %%
# normalize continuous variables
college_completion_encoded[['pell_percentile', 'grad_100_percentile','long_x','lat_y','med_sat_percentile']] = MinMaxScaler().fit_transform(college_completion_encoded[['pell_percentile', 'grad_100_percentile','long_x','lat_y','med_sat_percentile']])   

college_completion_encoded.head()

# %%
# split into train and test sets
train, test = train_test_split(college_completion_encoded, train_size=0.7, stratify=college_completion_encoded['hbcu_X'])

print(train.shape)
print(test.shape)

# %%
# split test set into tune and test sets
tune, test = train_test_split(test, train_size=0.5, stratify=test['hbcu_X'])

print(tune.shape)
print(test.shape)

# %%
features = ['lat_y', 'long_x', 'grad_100_percentile', 'pell_percentile', 'med_sat_percentile']
target = 'hbcu_X'

X_train = train[features]
y_train = train[target]

X_tune = tune[features]
y_tune = tune[target]

X_test = test[features]
y_test = test[target]

# %%
k = 3  

final_knn = KNeighborsClassifier(n_neighbors=k)
final_knn.fit(X_train, y_train)

pred_test = final_knn.predict(X_test)

cm = confusion_matrix(y_test, pred_test)

print("Confusion Matrix:")
print("TN:", cm[0,0])
print("FP:", cm[0,1])
print("FN:", cm[1,0])
print("TP:", cm[1,1])

# %%
test_prob = final_knn.predict_proba(X_test)[:, 1]

results_df = pd.DataFrame({
    "actual": y_test,
    "predicted": pred_test,
    "prob_positive": test_prob
})

print(results_df.head(20))

# %%
def clean_and_split_data(
    df,
    features,
    target_col,
    positive_level,
    train_size=0.7,
    tune_fraction_of_holdout=0.5,  
    random_state=0
):

    # keep only needed columns
    df = df[[target_col] + features].copy()

    # one-hot encode target to binary column named like hbcu_X
    df = pd.get_dummies(df, columns=[target_col])
    y_col = f"{target_col}_{positive_level}"
    df[y_col] = df[y_col].astype(int)

    # drop NAs
    df = df.dropna().copy()

    # scale features
    df[features] = MinMaxScaler().fit_transform(df[features])

    # split train / holdout
    train_df, holdout_df = train_test_split(
        df, train_size=train_size, stratify=df[y_col], random_state=random_state
    )

    # split holdout into tune / test
    tune_df, test_df = train_test_split(
        holdout_df, train_size=tune_fraction_of_holdout, stratify=holdout_df[y_col], random_state=random_state
    )

    X_train, y_train = train_df[features], train_df[y_col]
    X_tune, y_tune = tune_df[features], tune_df[y_col]
    X_test, y_test = test_df[features], test_df[y_col]

    return {
        "X_train": X_train, "y_train": y_train,
        "X_tune": X_tune, "y_tune": y_tune,
        "X_test": X_test, "y_test": y_test,
        "target_binary_col": y_col,
        "df_clean": df
    }

# %%
def train_and_test_knn_with_threshold(
    X_train, 
    y_train, 
    X_eval, 
    y_eval, 
    k=3, 
    threshold=0.5):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    # predicted probabilities for positive class (class = 1)
    prob_pos = model.predict_proba(X_eval)[:, 1]

    # apply threshold to get predicted labels
    pred = (prob_pos >= threshold).astype(int)

    cm = confusion_matrix(y_eval, pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    # metrics (use zero_division=0 so it doesn't crash if no positives predicted)
    acc = accuracy_score(y_eval, pred)
    prec = precision_score(y_eval, pred, zero_division=0)
    rec = recall_score(y_eval, pred, zero_division=0)
    f1 = f1_score(y_eval, pred, zero_division=0)

    return {
        "k": k,
        "threshold": threshold,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

# %%
features = ['lat_y', 'long_x', 'grad_100_percentile', 'pell_percentile', 'med_sat_percentile']
target = 'hbcu'

splits = clean_and_split_data(
    df = college_completion,
    features=features,
    target_col = target,
    positive_level="X",
    train_size=0.7,
    tune_fraction_of_holdout=0.5,
    random_state=0
)

k_values = list(range(1,51,2))
threshold_values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

rows = []
for k in k_values:
    for t in threshold_values:
        out = train_and_test_knn_with_threshold(
            splits["X_train"], splits["y_train"],
            splits["X_tune"], splits["y_tune"],
            k=k, threshold=t
        )
        rows.append(out)

results = pd.DataFrame(rows)

# Pick best on tune by F1 (ties broken by higher recall, then higher precision).
best_tune = results.sort_values(["f1", "recall", "precision", "accuracy"], ascending=False).head(10)
print("Top 10 (tune set):")
print(best_tune)

best_k = int(best_tune.iloc[0]["k"])
best_t = float(best_tune.iloc[0]["threshold"])

print("\nBest (tune) choice:")
print("k =", best_k, "| threshold =", best_t)

# Final evaluation on TEST using the best (k, threshold)
final_test = train_and_test_knn_with_threshold(
    splits["X_train"], splits["y_train"],
    splits["X_test"], splits["y_test"],
    k=best_k, threshold=best_t
)

print("\nFinal TEST performance with best (k, threshold):")
print(final_test)

# %%
college_completion['grad_rate_high'] = (college_completion['grad_100_percentile'] >= 75.0).astype(int)
college_completion[['grad_100_percentile', 'grad_rate_high']].head(20)

# %%
target = 'grad_rate_high'
features = ['endow_percentile', 'pell_percentile', 'med_sat_percentile',]

splits = clean_and_split_data(target_col=target, features=features, positive_level=1, df = college_completion)

rows = []
for k in k_values:
    for t in threshold_values:
        out = train_and_test_knn_with_threshold(
            splits["X_train"], splits["y_train"],
            splits["X_tune"], splits["y_tune"],
            k=k, threshold=t
        )
        rows.append(out)

results = pd.DataFrame(rows)


best_tune = results.sort_values(["f1", "recall", "precision", "accuracy"], ascending=False).head(10)
print("Top 10 (tune set):")
print(best_tune)

best_k = int(best_tune.iloc[0]["k"])
best_t = float(best_tune.iloc[0]["threshold"])

print("\nBest (tune) choice:")
print("k =", best_k, "| threshold =", best_t)

# Final evaluation on TEST using the best (k, threshold)
final_test = train_and_test_knn_with_threshold(
    splits["X_train"], splits["y_train"],
    splits["X_test"], splits["y_test"],
    k=best_k, threshold=best_t
)

print("\nFinal TEST performance with best (k, threshold):")
print(final_test)

# %%
# QUESTIONS

# changing k changes the spacing for possible probability values, as they are all multiples of 1/k.
# larger k means the threshold function is smoother, smaller k means it is more abrupt.
# changing k changes the predicted probabilities, so appying the same threshold will produce different classifications, so the confusion will look different.

# one positive element about my model is it is very accurate. with k = 3, my overal accuracy was around 98%, which is great.
# one concern I have is that the baseline accruacy for the dataset was already around 96%, so the model is only slightly improving accuracy.

# the model performs execptionally well, with an accuracy of 97.47%. 
# changing the threshold and k values helped find the optimal combination to make the model as accurate as possible.
# this is because adjusting these factors changes how aggressively the model predicts a positive class.