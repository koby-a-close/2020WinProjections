# Predictions_GBR_v1.py
# Created 01/13/2020 by KAC

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
# Batting Stats: Runs Scored, wOBA, HRs, WAR
df_input['runsScored2017'] = batting_2017.R
df_input['wOBA2017'] = batting_2017.wOBA
df_input['WAR2017'] = batting_2017.WAR
df_input['runsScored2018'] = batting_2018.R
df_input['wOBA2018'] = batting_2018.wOBA
df_input['WAR2018'] = batting_2018.WAR
df_input['WAR2019'] = batting_2019.WAR

# Fielding Stats: DRS, Defense
df_input['DRS2017'] = fielding_2017.DRS
df_input['Def2017'] = fielding_2017.Def
df_input['DRS2018'] = fielding_2018.DRS
df_input['Def2018'] = fielding_2018.Def
# Pitching Stats: FIP, xFIP, WAR
df_input['FIP2017'] = pitching_2017.FIP
df_input['xFIP2017'] = pitching_2017.xFIP
df_input['pWAR2017'] = pitching_2017.WAR
df_input['FIP2018'] = pitching_2018.FIP
df_input['xFIP2018'] = pitching_2018.xFIP
df_input['pWAR2018'] = pitching_2018.WAR
df_input['pWAR2019'] = pitching_2019.WAR

# TODO: Try a grid search to look for features/parameters of importance.
# Started with the previous season's wins. Building in more factors to decrease average error.
X = df_input[['wins2018','runsScored2018','wOBA2018', 'FIP2018', 'runsScored2017',  'wOBA2017', 'WAR2017', 'WAR2018',
              'DRS2017', 'DRS2018', 'Def2017', 'Def2018', 'FIP2017', 'xFIP2017', 'xFIP2018', 'pWAR2017', 'pWAR2018',
              'wins2017']]
# X = df_input[['wins2018','runsScored2018','wOBA2018', 'FIP2018']]

y = df_input[['wins2019']]

low_ci_model = GradientBoostingRegressor(loss='quantile', alpha=0.1).fit(X, y)
mean_ci_model = GradientBoostingRegressor(loss='quantile', alpha=0.5).fit(X, y)
high_ci_model = GradientBoostingRegressor(loss='quantile', alpha=0.9).fit(X, y)

print(pd.DataFrame({'Variable':X.columns,
              'Importance':mean_ci_model.feature_importances_}).sort_values('Importance', ascending=False))

# Make predictions for 2019 wins with 90% confidence interval
low_ci = low_ci_model.predict(X)
mean_ci = mean_ci_model.predict(X)
high_ci = high_ci_model.predict(X)

win_range = pd.DataFrame()
win_range['Team'] = team_records.Team
win_range['mean_ci_lower'] = low_ci.T
win_range['mean'] = mean_ci
win_range['mean_ci_upper'] = high_ci
win_range['actual'] = team_records.wins2019

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
# Batting Stats: Runs Scored, wOBA, HRs, WAR
df_pred_input['runsScored2019'] = batting_2019.R
df_pred_input['wOBA2019'] = batting_2019.wOBA
df_pred_input['WAR2019'] = batting_2019.WAR
df_pred_input['runsScored2018'] = batting_2018.R
df_pred_input['wOBA2018'] = batting_2018.wOBA
df_pred_input['WAR2018'] = batting_2018.WAR
# Fielding Stats: DRS, Defense
df_pred_input['DRS2019'] = fielding_2019.DRS
df_pred_input['Def2019'] = fielding_2019.Def
df_pred_input['DRS2018'] = fielding_2018.DRS
df_pred_input['Def2018'] = fielding_2018.Def
# Pitching Stats: FIP, xFIP, WAR
df_pred_input['FIP2019'] = pitching_2019.FIP
df_pred_input['xFIP2019'] = pitching_2019.xFIP
df_pred_input['pWAR2019'] = pitching_2019.WAR
df_pred_input['FIP2018'] = pitching_2018.FIP
df_pred_input['xFIP2018'] = pitching_2018.xFIP
df_pred_input['pWAR2018'] = pitching_2018.WAR

df_pred_input['WAR2020'] = team_proj.WAR
df_pred_input['pWAR2020'] = team_proj.pWAR


X_2020 = df_pred_input[['wins2018', 'wins2019', 'runsScored2019', 'runsScored2018', 'wOBA2019', 'wOBA2018', 'WAR2019',
                   'WAR2018','DRS2019', 'DRS2018', 'Def2019', 'Def2018', 'FIP2019', 'FIP2018', 'xFIP2019', 'xFIP2018',
                   'pWAR2019','pWAR2018']]

low_ci_2020 = low_ci_model.predict(X_2020)
mean_ci_2020 = mean_ci_model.predict(X_2020)
high_ci_2020 = high_ci_model.predict(X_2020)

win_range_2020 = pd.DataFrame()
win_range_2020['Team'] = team_records.Team
win_range_2020['mean_ci_lower'] = low_ci_2020
win_range_2020['mean'] = mean_ci_2020
win_range_2020['mean_ci_upper'] = high_ci_2020

GBR_v1 = pd.DataFrame()
GBR_v1['Team'] = team_records.Team
GBR_v1['Wins'] = round(win_range_2020['mean'], 1)
print(GBR_v1.head(30))

x=1

