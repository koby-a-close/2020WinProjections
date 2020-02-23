def MLR_ModelBuilder(X_train, y_train, X_test, y_test, X_unknown):

    # MLR_ModelBuilder.py
    # Created by KAC on 02/11/2020

    """ This function takes in data and completes a grid search to tune parameters automatically. It then makes predictions
    and calculates an MAE score for those predictions."""
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    import pandas as pd
    from sklearn.feature_selection import RFECV
    from sklearn.metrics import mean_absolute_error
    from sklearn.linear_model import LinearRegression as MLR
    from sklearn.model_selection import cross_val_score, RandomizedSearchCV
    from sklearn.metrics import make_scorer

    scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    MLR_model = MLR()
    RFECV = RFECV(estimator=MLR_model, scoring=scorer)
    RFECV.fit(X_train, y_train)
    CV_score = cross_val_score(RFECV, X_train, y_train, scoring=scorer)
    scr = np.mean(CV_score)
    print(pd.DataFrame({'Variable':X_train.columns,
                  'Importance':RFECV.ranking_}).sort_values('Importance', ascending=True).head(50))
    print("Optimal number of features: ", RFECV.n_features_)
    print("MAE for All Features on Training Data: ", scr)

    X_train_transformed = RFECV.transform(X_train)
    X_test_transformed = RFECV.transform(X_test)

    CV_score = cross_val_score(RFECV, X_train_transformed, y_train, scoring=scorer)
    scr = np.mean(CV_score)
    print("MAE for Selected Features on Training Data: ", scr)

    parameters = {'fit_intercept': [True, False], 'normalize': [True, False], 'copy_X': [True, False]}
    rsearch = RandomizedSearchCV(estimator=MLR_model, param_distributions=parameters, n_iter=100)
    rsearch.fit(X_train_transformed, y_train)
    # print(rsearch.best_params_)

    CV_score = cross_val_score(rsearch, X_train_transformed, y_train, scoring=scorer)
    scr = np.mean(CV_score)
    print("MAE for Selected Features and Parameter Tuning on Training Data: ", scr)

    predictions = rsearch.predict(X_test_transformed)
    pred_scr = round(mean_absolute_error(y_test, predictions), 3)
    print("MAE for Selected Features and Parameter Tuning on 2019 Data: ", pred_scr)

    if X_unknown is not None:
        X_final = pd.concat([X_train, X_test])
        X_final = RFECV.transform(X_final)
        y_final = pd.concat([y_train, y_test])

        X_unknown = RFECV.transform(X_unknown)

        rsearch.fit(X_final, y_final)
        predictions_final = rsearch.predict(X_unknown)

    else:
        predictions_final = []

    return predictions, predictions_final
