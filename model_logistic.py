import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from config import RANDOM_STATE

def train_logistic(X_train, y_train):
    model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name='Modèle'):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"--- Évaluation {model_name} ---")
    print(classification_report(y_test, predictions, digits=4))
    print("AUC:", roc_auc_score(y_test, probabilities))
    print("Matrice de confusion:")
    print(confusion_matrix(y_test, predictions))

    return {
        'auc': roc_auc_score(y_test, probabilities),
        'confusion_matrix': confusion_matrix(y_test, predictions)
    }


def plot_roc_curves(models, X_test, y_test):
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        probabilities = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probabilities)
        auc_value = roc_auc_score(y_test, probabilities)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_value:.3f})")

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Hasard')
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbes ROC comparées')
    plt.legend(loc='lower right')
    plt.grid(True)

    return plt
