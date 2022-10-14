import os

import numpy as np
import pandas as pd
import plotly.express
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import balanced_accuracy_score, mean_squared_error
from sklearn.ensemble import HistGradientBoostingClassifier

class CustomPipeline(Pipeline):
    #CHECK TO REPLACE STANDARDSCALER
    def partial_fit(self, x, y, classes,  weights=None):
        # x = self.named_steps.scaler.partial_fit(x)

        # cat_subset = x.select_dtypes(include=['object', 'category', 'bool'])
        # categorical_values = []

        # for i in range(cat_subset.shape[1]):
        #     categorical_values.append(list(cat_subset.iloc[:, i].dropna().unique()))

        # num_pipeline = Pipeline([
        #     ('cleaner', SimpleImputer()),
        #     ('scaler', StandardScaler()),
        #     # ('variance', VarianceThreshold())
        # ])
        #
        # cat_pipeline = CustomPipeline([
        #     ('cleaner', SimpleImputer(strategy='most_frequent')),
        #     ('encoder', OneHotEncoder(sparse=False, categories=categorical_values))
        # ])
        #
        # preprocessor = ColumnTransformer([
        #     ('numerical', num_pipeline, make_column_selector(dtype_exclude=['object', 'category', 'bool'])),
        #     ('categorical', cat_pipeline, make_column_selector(dtype_include=['object', 'category', 'bool']))
        # ])

        # self.named_steps.preprocessor = preprocessor
        # x = self.named_steps.preprocessor.fit_transform(x)
        # x = self.named_steps.feature_selector.fit_transform(x, y)
        self.named_steps.scaler.partial_fit(x)
        x = self.named_steps.scaler.transform(x)
        if 'MLP' in str(self.named_steps.estimator):
            self.named_steps.estimator.partial_fit(x, y, classes=classes)
        else:
            self.named_steps.estimator.partial_fit(x, y, classes=classes, sample_weight=weights)
        # self._final_estimator = self.named_steps.estimator
        # self.named_steps.estimator.partial_fit(x,y, classes=np.unique(classes))



class MyAutoMLClassifier:
    def __init__(self, scoring_function='balanced_accuracy', n_iter=10):
        self.scoring_function = scoring_function
        self.n_iter = n_iter

    def fit(self, X, y):

        X_train = X
        y_train = y

        # constant_filter = VarianceThreshold(threshold=0)
        # constant_filter.fit(X_train)
        # constant_columns = [column for column in X_train.columns
        #                     if column not in
        #                     X_train.columns[constant_filter.get_support()]]
        # X_train = constant_filter.transform(X_train)
        # for column in constant_columns:
        #     print("REMOVED: ", column)

        categorical_values = []
        cat_subset = X_train.select_dtypes(include=['object', 'category', 'bool'])

        for i in range(cat_subset.shape[1]):
            categorical_values.append(list(cat_subset.iloc[:, i].dropna().unique()))

        num_pipeline = Pipeline([
            ('cleaner', SimpleImputer()),
            ('scaler', StandardScaler()),
            # ('variance', VarianceThreshold())
        ])

        cat_pipeline = CustomPipeline([
            ('cleaner', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(sparse=False, categories=categorical_values))
        ])

        preprocessor = ColumnTransformer([
            ('numerical', num_pipeline, make_column_selector(dtype_exclude=['object', 'category', 'bool'])),
            ('categorical', cat_pipeline, make_column_selector(dtype_include=['object', 'category', 'bool']))
        ])

        model_pipeline_steps = []
        model_pipeline_steps.append(('preprocessor', preprocessor))
        model_pipeline_steps.append(('feature_selector', SelectKBest(f_classif, k='all')))
        model_pipeline_steps.append(('estimator', LogisticRegression()))
        model_pipeline = CustomPipeline(model_pipeline_steps)

        total_features = preprocessor.fit_transform(X_train).shape[1]

        optimization_grid = []

        # Logistic regression
        # optimization_grid.append({
        #     'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
        #     'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
        #     'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
        #     'estimator': [LogisticRegression()]
        # })

        # K-nearest neighbors
        # optimization_grid.append({
        #     'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
        #     'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
        #     'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
        #     'estimator': [KNeighborsClassifier()],
        #     'estimator__weights': ['uniform', 'distance'],
        #     'estimator__n_neighbors': np.arange(1, 20, 1)
        # })

        # Random Forest
        # optimization_grid.append({
        #     'preprocessor__numerical__scaler': [None],
        #     'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
        #     'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
        #     'estimator': [RandomForestClassifier(random_state=0)],
        #     'estimator__n_estimators': np.arange(5, 500, 10),
        #     'estimator__criterion': ['gini', 'entropy']
        # })

        # Gradient boosting
        # optimization_grid.append({
        #     'preprocessor__numerical__scaler': [None],
        #     'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
        #     'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
        #     'estimator': [GradientBoostingClassifier(random_state=0, warm_start=True)],
        #     'estimator__n_estimators': np.arange(5, 500, 10),
        #     'estimator__learning_rate': np.linspace(0.1, 0.9, 5),
        #     'estimator__subsample': np.linspace(0.7, 0.9, 3)
        #
        # })

        # Gradient boosting Hystogrram
        optimization_grid.append({
            'preprocessor__numerical__scaler': [None, StandardScaler()],
            'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
            'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
            'estimator': [HistGradientBoostingClassifier(random_state=0, warm_start=True)],
            'estimator__learning_rate': np.linspace(0.1, 0.9, 5)

        })

        # Decision tree
        # optimization_grid.append({
        #     'preprocessor__numerical__scaler': [None],
        #     'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
        #     'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
        #     'estimator': [DecisionTreeClassifier(random_state=0)],
        #     'estimator__criterion': ['gini', 'entropy']
        # })

        # Linear SVM
        # optimization_grid.append({
        #     'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
        #     'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
        #     'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
        #     'estimator': [LinearSVC(random_state=0)],
        #     'estimator__C': np.arange(0.1, 1, 0.1),
        #
        # })

        # SGD
        # optimization_grid.append({
        #     'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
        #     # 'preprocessor__numerical__variance__threshold': [0, 0.1, 0.2],
        #     'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
        #     'feature_selector__k': list(np.arange(1, total_features, 1)) + ['all'],
        #     'estimator': [SGDClassifier(random_state=0, warm_start=True, learning_rate='adaptive', shuffle=False)],
        #     'estimator__penalty': ['l2', 'l1'],
        #     'estimator__loss': ['hinge', 'huber', 'modified_huber'],
        #     "estimator__alpha": [0.0001, 0.001, 0.01, 0.1],
        #     "estimator__eta0": [0.01, 0.1, 0.3]
        # })

        search = RandomizedSearchCV(
            model_pipeline,
            optimization_grid,
            n_iter=self.n_iter,
            scoring=self.scoring_function,
            n_jobs=os.cpu_count()//2,
            random_state=0,
            verbose=3,
            # cv=PredefinedSplit(test_fold=[0 for _ in validation_data.index.__iter__()])
            cv=3,
        )

        search.fit(X_train, y_train)
        self.best_estimator_ = search.best_estimator_
        self.best_pipeline = search.best_params_


    def predict(self, X, y=None):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X, y=None):
        return self.best_estimator_.predict_proba(X)


def get_best_classifier(df):
    X, y = select_x_y_for_training(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    # X_val, Y_val = select_x_y_for_training(tail_df)
    # _, X_val, _, Y_val = train_test_split(X_val, Y_val, test_size=len(tail_df)-1, random_state=42)
    model = MyAutoMLClassifier()
    model.fit(X_train, y_train)
    get_model_accuracy(X_test, y_test, model)
    model.best_pipeline

    return model


def get_model_accuracy(x_test, y_test, model):
    accuracy = balanced_accuracy_score(y_test.apply(pd.to_numeric), model.predict(x_test))
    print(f"%%% - MODEL BALANCED ACCURACY - %%%: {accuracy}")
    return accuracy


def select_x_y_for_training(mydf, second_df=None):
    column_condition =  [col for col in mydf.columns
                             if 'timestamp' not in col
                             and 'others_cr' not in col
                             and 'BTC_' not in col
                                and 'unix' not in col
                                and 'symbol' not in col
                                and 'Volume' not in col
                                and 'bb_' not in col
                                ]

    # mydf[column_condition].dropna(inplace=True, how='any', axis=0)
    y = mydf['BTC_index']
    X = pd.DataFrame(mydf[
                         column_condition
                     ], columns=column_condition
                     )
    return X, y


def split_dataset(x, y):
    x_train, x_test, y_train_, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    return x_train, x_test, y_train_, y_test
