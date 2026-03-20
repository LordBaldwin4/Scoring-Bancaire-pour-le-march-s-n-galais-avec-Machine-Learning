from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import TEST_SIZE, RANDOM_STATE

def preprocess(data):
    # on sépare les variables explicatives (ce qu'on donne au modèle)
    # de la cible (ce qu'on veut qu'il prédise)
    X = data.drop('Defaut', axis=1)
    y = data['Defaut']

    # on découpe en deux parties : une pour entraîner le modèle, une pour le tester
    # le modèle ne verra jamais les données de test pendant l'entraînement
    # random_state assure qu'on obtient le même découpage à chaque fois
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,       # proportion des données réservée au test (définie dans config.py)
        random_state=RANDOM_STATE
    )

    # le revenu mensuel est en millions de FCFA, l'âge en années, la durée en mois...
    # sans standardisation, le modèle serait biaisé vers les grandes valeurs
    # StandardScaler ramène tout à moyenne=0 et écart-type=1
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # on apprend l'échelle sur les données d'entraînement...
    X_test  = scaler.transform(X_test)       # ...et on l'applique au test sans le recalculer — règle importante

    # on retourne aussi les noms des colonnes pour pouvoir les afficher dans les graphiques plus tard
    return X_train, X_test, y_train, y_test, X.columns
