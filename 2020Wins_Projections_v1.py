# 2020Wins_Predictions_v1.py
# Created 02/10/2020 by KAC

""" The model works to predict 2020 win totals for all 30 MLB teams.
Multiple model types will be tried: MLR, GBR, SVM
Feature selection will be done using RFE.
Model performance will be evaluated using MAE.
The final model will be tuned using a grid search for features.
The final model will also be compared to Fan Duel O/U lines."""

# Load packages
import pandas as pd
from MLR_ModelBuilder import MLR_ModelBuilder
from XGB_ModelBuilder import XGB_ModelBuilder
from SVM_ModelBuilder import SVM_ModelBuilder
from LOG_ModelBuilder import LOG_ModelBuilder
from sklearn import preprocessing

from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression as MLR
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

# IMPORT DATA
data_dir = '/Users/Koby/PycharmProjects/2020RecordPredictions/Input/'

batting_2019 = pd.read_csv(data_dir + 'Batting_2019.csv')
batting_2018 = pd.read_csv(data_dir + 'Batting_2018.csv')
batting_2017 = pd.read_csv(data_dir + 'Batting_2017.csv')
batting_2016 = pd.read_csv(data_dir + 'Batting_2016.csv')
batting_2015 = pd.read_csv(data_dir + 'Batting_2015.csv')

batting_2019 = batting_2019[['Team', 'AVG', 'OBP', 'SLG', 'wOBA', 'Off', 'BsR', 'Def', 'WAR']]
batting_2018 = batting_2018[['Team', 'AVG', 'OBP', 'SLG', 'wOBA', 'Off', 'BsR', 'Def', 'WAR']]
batting_2017 = batting_2017[['Team', 'AVG', 'OBP', 'SLG', 'wOBA', 'Off', 'BsR', 'Def', 'WAR']]
batting_2016 = batting_2016[['Team', 'AVG', 'OBP', 'SLG', 'wOBA', 'Off', 'BsR', 'Def', 'WAR']]
batting_2015 = batting_2015[['Team', 'AVG', 'OBP', 'SLG', 'wOBA', 'Off', 'BsR', 'Def', 'WAR']]

pitching_2019 = pd.read_csv(data_dir + 'Pitching_2019.csv')
pitching_2018 = pd.read_csv(data_dir + 'Pitching_2018.csv')
pitching_2017 = pd.read_csv(data_dir + 'Pitching_2017.csv')
pitching_2016 = pd.read_csv(data_dir + 'Pitching_2016.csv')
pitching_2015 = pd.read_csv(data_dir + 'Pitching_2015.csv')

pitching_2019 = pitching_2019[['Team', 'K/9', 'BB/9', 'HR/9', 'BABIP', 'LOB', 'ERA', 'FIP', 'WAR']]
pitching_2018 = pitching_2018[['Team', 'K/9', 'BB/9', 'HR/9', 'BABIP', 'LOB', 'ERA', 'FIP', 'WAR']]
pitching_2017 = pitching_2017[['Team', 'K/9', 'BB/9', 'HR/9', 'BABIP', 'LOB', 'ERA', 'FIP', 'WAR']]
pitching_2016 = pitching_2016[['Team', 'K/9', 'BB/9', 'HR/9', 'BABIP', 'LOB', 'ERA', 'FIP', 'WAR']]
pitching_2015 = pitching_2015[['Team', 'K/9', 'BB/9', 'HR/9', 'BABIP', 'LOB', 'ERA', 'FIP', 'WAR']]

team_records = pd.read_csv(data_dir + 'team_records.csv')
projections_2020 = pd.read_csv(data_dir + 'WAR_proj_2020.csv')

# COMBINE ALL DATA INTO A SINGLE DF
X_20 = batting_2019.merge(pitching_2019, on='Team')
X_20 = X_20.add_suffix('_prev')
X_20['wins_prev'] = team_records.wins2019
X_20 = X_20.merge(projections_2020, left_on='Team_prev', right_on='Team')
X_20 = X_20.rename(columns={'WAR_x_prev': 'WAR_prev', 'WAR_y_prev': 'pWAR_prev'})
X_20 = X_20.drop(columns=['Team_prev','Team', 'PA', 'IP'])
X_20 = X_20.dropna()

X_19 = batting_2018.merge(pitching_2018, on='Team')
X_19 = X_19.add_suffix('_prev')
X_19['wins_prev'] = team_records.wins2018
X_19 = X_19.merge(batting_2019, left_on='Team_prev', right_on='Team')
X_19 = X_19.merge(pitching_2019, on='Team')
X_19 = X_19.rename(columns={'WAR_x_prev': 'WAR_prev', 'WAR_y_prev': 'pWAR_prev', 'WAR_x': 'WAR', 'WAR_y': 'pWAR'})
X_19 = X_19.drop(columns=['Team_prev','Team'])
y_19 = team_records['wins2019']
X_19 = X_19.dropna()
y_19 = y_19.dropna()

X_18 = batting_2017.merge(pitching_2017, on='Team')
X_18 = X_18.add_suffix('_prev')
X_18['wins_prev'] = team_records.wins2017
X_18 = X_18.merge(batting_2018, left_on='Team_prev', right_on='Team')
X_18 = X_18.merge(pitching_2018, on='Team')
X_18 = X_18.rename(columns={'WAR_x_prev': 'WAR_prev', 'WAR_y_prev': 'pWAR_prev', 'WAR_x': 'WAR', 'WAR_y': 'pWAR'})
X_18 = X_18.drop(columns=['Team_prev','Team'])
y_18 = team_records['wins2018']

X_17 = batting_2016.merge(pitching_2016, on='Team')
X_17 = X_17.add_suffix('_prev')
X_17['wins_prev'] = team_records.wins2016
X_17 = X_17.merge(batting_2017, left_on='Team_prev', right_on='Team')
X_17 = X_17.merge(pitching_2017, on='Team')
X_17 = X_17.rename(columns={'WAR_x_prev': 'WAR_prev', 'WAR_y_prev': 'pWAR_prev', 'WAR_x': 'WAR', 'WAR_y': 'pWAR'})
X_17 = X_17.drop(columns=['Team_prev','Team'])
y_17 = team_records['wins2017']

X_16 = batting_2015.merge(pitching_2015, on='Team')
X_16 = X_16.add_suffix('_prev')
X_16['wins_prev'] = team_records.wins2015
X_16 = X_16.merge(batting_2016, left_on='Team_prev', right_on='Team')
X_16 = X_16.merge(pitching_2016, on='Team')
X_16 = X_16.rename(columns={'WAR_x_prev': 'WAR_prev', 'WAR_y_prev': 'pWAR_prev', 'WAR_x': 'WAR', 'WAR_y': 'pWAR'})
X_16 = X_16.drop(columns=['Team_prev','Team'])
y_16 = team_records['wins2016']

X = pd.concat([X_18, X_17, X_16], ignore_index=True)
y = pd.concat([y_18, y_17, y_16], ignore_index=True)

# Check to see if there are any empty values
# idx, idy = np.where(pd.isnull(X))
# result = np.column_stack([X.index[idx], X.columns[idy]])
# print(result)

X = X.dropna()
y = y.dropna()

# MLR Model
print("MLR MODEL SUMMARY:")
MLR_predictions_2019, MLR_predictions_2020 = MLR_ModelBuilder(X, y, X_19, y_19, X_20)

MLR_pred_2019 = pd.DataFrame()
MLR_pred_2019['Team'] = team_records.Team
MLR_pred_2019['Predicted Wins'] = MLR_predictions_2019
MLR_pred_2019['Actual Wins'] = team_records.wins2019

MLR_pred_2020 = pd.DataFrame()
MLR_pred_2020['Team'] = team_records.Team
MLR_pred_2020['Projected 2020 Wins'] = MLR_predictions_2020

print(sum(MLR_pred_2019['Predicted Wins']))
print(sum(MLR_pred_2020['Projected 2020 Wins']))


# # XGBoost Model
# print("XGB MODEL SUMMARY:")
# XGB_predictions_2019, XGB_predictions_2020 = XGB_ModelBuilder(X, y, X_19, y_19, X_20)
#
# XGB_pred_2019 = pd.DataFrame()
# XGB_pred_2019['Team'] = team_records.Team
# XGB_pred_2019['Predicted Wins'] = XGB_predictions_2019
# XGB_pred_2019['Actual Wins'] = team_records.wins2019
#
# XGB_pred_2020 = pd.DataFrame()
# XGB_pred_2020['Team'] = team_records.Team
# XGB_pred_2020['Predicted Wins'] = XGB_predictions_2020
#
# # SVM Model: Very poor performance - SKIP
# print("SVM MODEL SUMMARY:")
# SVM_predictions_2019, SVM_predictions_2020 = SVM_ModelBuilder(X, y, X_19, y_19, X_20)
#
# SVM_pred_2019 = pd.DataFrame()
# SVM_pred_2019['Team'] = team_records.Team
# SVM_pred_2019['Predicted Wins'] = SVM_predictions_2019
# SVM_pred_2019['Actual Wins'] = team_records.wins2019
#
# SVM_pred_2020 = pd.DataFrame()
# SVM_pred_2020['Team'] = team_records.Team
# SVM_pred_2020['Predicted Wins'] = SVM_predictions_2020
#
# # Logistic Regression Model:
# print("LOG MODEL SUMMARY:")
# LOG_predictions_2019, LOG_predictions_2020 = LOG_ModelBuilder(X, y, X_19, y_19, X_20)
#
# LOG_pred_2019 = pd.DataFrame()
# LOG_pred_2019['Team'] = team_records.Team
# LOG_pred_2019['Predicted Wins'] = LOG_predictions_2019
# LOG_pred_2019['Actual Wins'] = team_records.wins2019
#
# LOG_pred_2020 = pd.DataFrame()
# LOG_pred_2020['Team'] = team_records.Team
# LOG_pred_2020['Predicted Wins'] = LOG_predictions_2020
#
# # TODO: Add more complex models using TensorFlow
# # TODO: Make ensemble predictions? Compare best model to FD O/U lines.


