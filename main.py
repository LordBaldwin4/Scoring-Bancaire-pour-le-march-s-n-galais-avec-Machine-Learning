from   generate_data_senegal import generate_data_senegal
from eda import plot_default_distribution
from preprocessing import preprocess
from model_logistic import train_logistic
from model_random_forest import train_random_forest
from evaluation import evaluate_model

def main():
    data = generate_data_senegal()

    plot_default_distribution(data)

    X_train, X_test, y_train, y_test, features = preprocess(data)

    print("\n--- Régression Logistique ---")
    lr_model = train_logistic(X_train, y_train)
    evaluate_model(lr_model, X_test, y_test)

    print("\n--- Random Forest ---")
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test)

if __name__ == "__main__":
    main()
