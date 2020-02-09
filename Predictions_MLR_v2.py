# Predictions_GBR_v2.py
# Created 02/06/2020 by KAC

""" Updates v1 to use three years of historical data."""
# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=Warning)

# Load packages
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from pybaseball import team_batting, team_pitching


# TODO: Research roster turnover in MLB and add more years. Also allow for multiple years to be tested for continued
#  model performance before applying to 2020.
# IMPORT DATA
data_dir = '/Users/Koby/PycharmProjects/2020RecordPredictions/Input/'
batting_2019 = pd.read_csv(data_dir + 'Batting_2019.csv')
batting_2018 = pd.read_csv(data_dir + 'Batting_2018.csv')
batting_2017 = pd.read_csv(data_dir + 'Batting_2017.csv')
batting_2016 = pd.read_csv(data_dir + 'Batting_2016.csv')
batting_2015 = pd.read_csv(data_dir + 'Batting_2015.csv')

pitching_2019 = pd.read_csv(data_dir + 'Pitching_2019.csv')
pitching_2018 = pd.read_csv(data_dir + 'Pitching_2018.csv')
pitching_2017 = pd.read_csv(data_dir + 'Pitching_2017.csv')
pitching_2016 = pd.read_csv(data_dir + 'Pitching_2016.csv')
pitching_2015 = pd.read_csv(data_dir + 'Pitching_2015.csv')


fielding_2019 = pd.read_csv(data_dir + 'Fielding_2019.csv')
fielding_2018 = pd.read_csv(data_dir + 'Fielding_2018.csv')
fielding_2017 = pd.read_csv(data_dir + 'Fielding_2017.csv')
fielding_2016 = pd.read_csv(data_dir + 'Fielding_2016.csv')
fielding_2015 = pd.read_csv(data_dir + 'Fielding_2015.csv')

team_records = pd.read_csv(data_dir + 'team_records.csv')
team_proj = pd.read_csv(data_dir + 'WAR_proj_2020.csv')

# TRAINING THE MODEL
# Training model to predict 2019 records using data from 2017 and 2018
# TODO: Add more elements to 'X' making the model more robust and accurate.
#  Would like to add lost WAR and added WAR in free agency as well as runs allowed
# Team Stats: Wins
df_input = team_records.copy()
# Batting WAR
df_input['WAR2015'] = batting_2015.WAR
df_input['WAR2016'] = batting_2016.WAR
df_input['WAR2017'] = batting_2017.WAR
df_input['WAR2018'] = batting_2018.WAR
df_input['WAR2019'] = batting_2019.WAR

# Pitching WAR (pWAR)
df_input['pWAR2015'] = pitching_2015.WAR
df_input['pWAR2016'] = pitching_2016.WAR
df_input['pWAR2017'] = pitching_2017.WAR
df_input['pWAR2018'] = pitching_2018.WAR
df_input['pWAR2019'] = pitching_2019.WAR

# TODO: Try a grid search to look for features/parameters of importance.
# Started with the previous season's wins. Building in more factors to decrease average error.
# X = df_input[['wins2018','runsScored2018','wOBA2018', 'FIP2018', 'runsScored2017',  'wOBA2017', 'WAR2017', 'WAR2018',
#               'DRS2017', 'DRS2018', 'Def2017', 'Def2018', 'FIP2017', 'xFIP2017', 'xFIP2018', 'pWAR2017', 'pWAR2018',
#               'wins2017']]
# X = df_input[['wins2018','runsScored2018','wOBA2018', 'FIP2018']]

X_19 = df_input[['wins2018', 'WAR2018', 'pWAR2018', 'WAR2019', 'pWAR2019']]
X_19 = X_19.rename(columns={'wins2018': 'prev_wins', 'WAR2018': 'prev_WAR', 'pWAR2018': 'prev_pWAR', 'WAR2019': 'WAR',
                            'pWAR2019': 'pWAR'})
y_19 = df_input['wins2019']

X_18 = df_input[['wins2017', 'WAR2017', 'pWAR2017', 'WAR2018', 'pWAR2018']]
X_18 = X_18.rename(columns={'wins2017': 'prev_wins', 'WAR2017': 'prev_WAR', 'pWAR2017': 'prev_pWAR', 'WAR2018': 'WAR',
                            'pWAR2018': 'pWAR'})
y_18 = df_input['wins2018']

X_17 = df_input[['wins2016', 'WAR2016', 'pWAR2016', 'WAR2017', 'pWAR2017']]
X_17 = X_17.rename(columns={'wins2016': 'prev_wins', 'WAR2016': 'prev_WAR', 'pWAR2016': 'prev_pWAR', 'WAR2017': 'WAR',
                            'pWAR2017': 'pWAR'})
y_17 = df_input['wins2017']

X_16 = df_input[['wins2015', 'WAR2015', 'pWAR2015', 'WAR2016', 'pWAR2016']]
X_16 = X_16.rename(columns={'wins2015': 'prev_wins', 'WAR2015': 'prev_WAR', 'pWAR2015': 'prev_pWAR', 'WAR2016': 'WAR',
                            'pWAR2016': 'pWAR'})
y_16 = df_input['wins2016']

X = pd.concat([X_19, X_18, X_17, X_16], ignore_index=True)
y = pd.concat([y_19, y_18, y_17, y_16], ignore_index=True)

results = sm.OLS(y, X).fit()
print(results.summary())

# Make predictions for 2019 wins with 80% confidence interval
predictions = results.get_prediction(X)
intervals = predictions.summary_frame(alpha=0.1)
intervals['team'] = team_records.Team

win_range = intervals[['team', 'mean_ci_lower', 'mean', 'mean_ci_upper']]
win_range.mean_ci_lower = round(win_range.mean_ci_lower, 1)
win_range['mean'] = round(win_range['mean'], 1)
win_range.mean_ci_upper = round(win_range.mean_ci_upper, 1)
win_range['actual'] = y

# EVALUATE MODEL PERFORMANCE
# Plots confidence interval and actual results. Goals is for all actual values to be within interval.
# NOT REALLY THAT VISUALLY PLEASING RIGHT NOW
plt.scatter(win_range.index, win_range.mean_ci_lower, c='black', marker='_')
plt.scatter(win_range.index, win_range.mean_ci_upper, c='black', marker='_')
plt.scatter(win_range.index, win_range['mean'], c='black', marker='.')
plt.scatter(win_range.index, win_range.actual, c='red', marker='x')
plt.show()

plt.scatter(win_range['mean'], win_range.actual)
plt.xlabel('Predicted Wins')
plt.ylabel('Actual Wins')
plt.show()

# Accuracy: how many actual win totals within CI range?
count = 0
for i in range(len(win_range)):
    if win_range.mean_ci_lower[i] < win_range.actual[i] < win_range.mean_ci_upper[i]:
        count += 1
print("Number of predictions within CI:", count)

# Average range of confidence interval and error in predictions for 2019
mean_range = round(np.mean(win_range.mean_ci_upper - win_range.mean_ci_lower), 1)
print("Average CI Range:", mean_range)
mean_error = round(mean_absolute_error(win_range.actual, win_range['mean']), 1)
print("Average prediction error:", mean_error)


# 2020 PREDICTIONS
# Team Stats: Wins
df_pred_input = team_records.copy()
# Batting WAR
df_pred_input['WAR2019'] = batting_2019.WAR

# Pitching WAR (pWAR)
df_pred_input['pWAR2019'] = pitching_2019.WAR

df_pred_input['WAR2020'] = team_proj.WAR
df_pred_input['pWAR2020'] = team_proj.pWAR


# X_2020 = df_pred_input[['wins2018', 'wins2019', 'runsScored2019', 'runsScored2018', 'wOBA2019', 'wOBA2018', 'WAR2019',
#                    'WAR2018','DRS2019', 'DRS2018', 'Def2019', 'Def2018', 'FIP2019', 'FIP2018', 'xFIP2019', 'xFIP2018',
#                    'pWAR2019','pWAR2018']]
X_2020 = df_pred_input[['wins2019', 'WAR2019', 'pWAR2019', 'WAR2020', 'pWAR2020']]

predictions_2020 = results.get_prediction(X_2020)
intervals_2020 = predictions_2020.summary_frame(alpha=0.1)
intervals_2020['team'] = team_records.Team

wins_2020 = intervals_2020[['team', 'mean_ci_lower', 'mean', 'mean_ci_upper']]
wins_2020.mean_ci_lower = round(wins_2020.mean_ci_lower, 1)
wins_2020['mean'] = round(wins_2020['mean'], 1)
wins_2020.mean_ci_upper = round(wins_2020.mean_ci_upper, 1)

MLR_v2 = pd.DataFrame()
MLR_v2['Team'] = team_records.Team
MLR_v2['Wins'] = round(wins_2020['mean'], 1)
print(MLR_v2.head(30))
MLR_v2.to_csv('MLR_v2_predictions.csv', index=False)

x=1

