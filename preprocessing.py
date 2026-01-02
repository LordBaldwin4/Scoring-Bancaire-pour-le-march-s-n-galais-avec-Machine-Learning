from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import TEST_SIZE, RANDOM_STATE

def preprocess(data):
    X = data.drop('Defaut', axis=1)
    y = data['Defaut']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X.columns
