# [1]
import sys
from urllib.request import urlopen
import numpy as np
import pandas as pd

# [2]
heartDisease_df = pd.read_csv('Datasets/heartDisease.csv')
# [3]
heartDisease_df.drop(['ca', 'slope', 'thal', 'oldpeak'], axis=1, inplace=True)
# [4]
heartDisease_df.replace('?', np.nan, inplace=True)
# [5]
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
model = BayesianModel([ ('age', 'trestbps'),
                        ('age', 'fbs'), ('sex', 'trestbps'),
                        ('exang', 'trestbps'), ('trestbps', 'heartdisease'),
                        ('fbs', 'heartdisease'), ('heartdisease', 'restecg'),
                        ('heartdisease', 'thalach'), ('heartdisease', 'chol')])
# Learing CPDs using Maximum Likelihood Estimators
model.fit(heartDisease_df, estimator=MaximumLikelihoodEstimator)

# [6]
print(model.get_cpds('age'))
print(model.get_cpds('chol'))
print(model.get_cpds('sex'))
model.get_independencies()

# [7]
from pgmpy.inference import VariableElimination
HeartDisease_infer = VariableElimination(model)
# [8]
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': 28})
print(q['heartdisease'])
# [9]
q = HeartDisease_infer.query(variables=['heartdisease'],evidence={'chol': 100})
print(q['heartdisease'])
