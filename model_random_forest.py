from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_random_forest(X_train, y_train):
    # on définit les combinaisons de paramètres qu'on veut tester
    # GridSearchCV va essayer toutes les combinaisons possibles et garder la meilleure
    param_grid = {
        'n_estimators':     [100, 200],        # nombre d'arbres dans la forêt — plus c'est grand, plus c'est stable
        'max_depth':        [5, 10, None],     # profondeur max de chaque arbre — None veut dire qu'on le laisse grandir librement
        'min_samples_split':[2, 5],            # nombre minimum de clients pour couper un nœud
        'min_samples_leaf': [1, 2]             # nombre minimum de clients dans une feuille finale
    }

    # on entraîne un Random Forest pour chaque combinaison de paramètres
    # cv=5 veut dire qu'on découpe les données en 5 parties et on teste sur chacune à tour de rôle
    # scoring='roc_auc' : on choisit le meilleur modèle selon l'AUC, pas juste la précision
    # c'est plus pertinent ici car on a un déséquilibre entre bons et mauvais clients
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),  # random_state=42 pour des résultats reproductibles
        param_grid,
        cv=5,
        scoring='roc_auc'
    )
    grid.fit(X_train, y_train)

    # on affiche les meilleurs paramètres trouvés pour garder une trace et pouvoir les réutiliser
    print("Best params RF:", grid.best_params_)

    # on retourne uniquement le meilleur modèle, pas toute la grille
    return grid.best_estimator_
