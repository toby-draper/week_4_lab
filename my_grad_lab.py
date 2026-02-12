# %%
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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
# # split test set into tune and test sets
# tune, test = train_test_split(test, train_size=0.5, stratify=test['hbcu_X'])

# print(tune.shape)
# print(test.shape)

# %%
features = ['lat_y', 'long_x', 'grad_100_percentile', 'pell_percentile', 'med_sat_percentile']
target = 'hbcu_X'

X_train = train[features]
y_train = train[target]

# X_tune = tune[features]
# y_tune = tune[target]

X_test = test[features]
y_test = test[target]s

# %%

k = 13  

final_knn = KNeighborsClassifier(n_neighbors=k)
final_knn.fit(X_train, y_train)

pred_test = final_knn.predict(X_test)

accuracy = accuracy_score(y_test, pred_test)

print(f"TEST accuracy: {accuracy * 100:.2f}%")

# %%
