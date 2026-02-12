# %% 
# Step One: Data Set Review and Question Formulationx
#
# College Completion
# This dataset contains information on the success and progress of college students in America. One question it could be used to address is comparing graduation rates between ethnicities.

# Job Placement
# This dataset contains information on former students and factors about their work experience. One question it could be used to address is comparing the salaries of male and female students in the same major type.

# %% 
# Step Two: Problem Definition
#
# College Completion: What are the differences in graduation rates between HBCUs and non-HBCUs?
# Independent Business Metric: HBCU
# 
# Job Placement: What are the differences in salaries across men and women of the same major?
# Independent Business Metric: Gender
# 
# Data Preparation Steps:

# %%
# import packages
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler, StandardScaler 

# set variables for dfs
college_completion = pd.read_csv("data/cc_institution_details.csv")
job_placement = pd.read_csv("data/Placement_Data_Full_Class 2.csv")

# %%
# examine college completion:
college_completion.info()

# %%
# drop unneeded columns
to_keep = [8,13,30,32]
college_completion_new = college_completion[college_completion.columns[to_keep]]

college_completion_new.info()

# %%
# transform hbcu values into binary using one hot encoding
college_completion_encoded = pd.get_dummies(college_completion_new, columns=['hbcu'])
college_completion_encoded["hbcu_X"] = (college_completion_encoded["hbcu_X"].astype(int))

college_completion_encoded.info()
college_completion_encoded.head()

# %%
# normalize continuous variables
# because graduation rate is a percentage, we don't need to scale it

# %%
# calculate hbcu prevalence
hbcu_prevalence = college_completion_encoded["hbcu_X"].mean() * 100
print(hbcu_prevalence)

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
# examine job placement:
job_placement.info()

# %%
# drop unneeded columns
to_keep = [1,8,14]
job_placement_new = job_placement[job_placement.columns[to_keep]]

job_placement_new.info()

# %%
# transform hbcu values into binary using one hot encoding
job_placement_encoded = pd.get_dummies(job_placement_new, columns = ['gender'])
job_placement_encoded['gender_M'] = (job_placement_encoded['gender_M'].astype(int))

job_placement_encoded.info()
job_placement_encoded.head()

# %%
# normalize continuous variables
job_placement_encoded['salary'] = MinMaxScaler().fit_transform(job_placement_encoded[['salary']])   

job_placement_encoded.head()

# %%
# calculate male gender prevalence
gender_prevalence = job_placement_encoded['gender_M'].mean() * 100
print(gender_prevalence)

# %%
# split into train and test sets
train, test = train_test_split(job_placement_encoded, train_size=0.7, stratify=job_placement_encoded['gender_M'])

print(train.shape)
print(test.shape)

# %%
# split test set into tune and test sets
tune, test = train_test_split(test, train_size=0.5, stratify = test['gender_M'])

print(tune.shape)
print(test.shape)

# %%
# Step Three: Data Assessment

# College Completion:


# 1) Can this dataset realistically address your problem?

# Realistically, this datset can address my problem from a purely analytical perspective, as I can use the data to compare the graduation rates of HBCUs and non-HBCUs


# 2) What areas or variables concern you?

# The main area that concerns me is the fact that the hbcu prevalence in the dataset is around 2-3%, so any model would have to be almost perfect to be useful.


# 3) What limitations or risks do you see?

# The main limitations of this dataset aren't really limitations at all. The first is that the prevalence of hbcus is so low that
# the baseline accuracy of the model is around 97%. This isn't a limitation by any means, but it makes building a useful model much harder.

# The second limitation is more of a user error on my end. In the context of this class, we have mostly been talking about kNN, a predicitive model.
# The problem I chose to examine isn't really one that a model can predict, so any model would be an analytical tool instead of a predictive one.



# Job Placement:

# 1) Can this dataset realistically address your problem?

# This dataset can be used to address my problem from both analytical and predictive perspectives. I can use the data to compare the salaries of men and women in the same degree type,
# or build a model to predict salary based on gender and degree type.


# 2) What areas or variables concern you?

# The main variable that concerns me is the degree type variable. There are only 3 unique values within the feature, meaning each category is likely very broad. This could lead to an 
# illusion of equivalence where none exists if women and men trend towards different majors within the same type of degree..


# 3) What limitations or risks do you see?

#) The main limitation I see in this dataset is the size. It is very small, and only contains data from one university. This could lead to false conclusions about the true relationship between
#) the earnings of men and women in similar fields.
