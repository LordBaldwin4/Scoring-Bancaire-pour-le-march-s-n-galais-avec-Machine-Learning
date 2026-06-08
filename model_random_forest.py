from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from config import RANDOM_STATE

def train_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators':     [100, 200],
        'max_depth':        [5, 8, 12],
        'min_samples_split':[5, 10],
        'min_samples_leaf': [2, 4, 6]
    }

    grid = GridSearchCV(
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1
        ),
        param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    grid.fit(X_train, y_train)

    print("Best params RF:", grid.best_params_)
    return grid.best_estimator_
