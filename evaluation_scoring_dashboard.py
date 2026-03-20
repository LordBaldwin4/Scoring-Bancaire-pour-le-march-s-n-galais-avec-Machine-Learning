# evaluation_scoring_dashboard.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -----------------------------
# CONFIG
# -----------------------------
RANDOM_STATE = 42  # graine fixe pour avoir les mêmes résultats à chaque exécution
sns.set_theme(style="darkgrid")
PALETTE = {'0': "#1f77b4", '1': "#ff7f0e"}  # bleu pour les bons clients, orange pour les défauts

# -----------------------------
# Génération dataset
# -----------------------------
def generate_data_senegal(n_samples=3000):
    np.random.seed(RANDOM_STATE)

    # on fabrique 3000 clients fictifs avec des caractéristiques typiques du contexte sénégalais
    # les revenus sont en FCFA, les montants de crédit aussi
    df = pd.DataFrame({
        'Age':                      np.random.randint(18, 65, n_samples),
        'Revenu_Mensuel':           np.random.randint(50000, 1500000, n_samples),
        'Anciennete_Emploi':        np.random.randint(0, 35, n_samples),
        'Montant_Credit':           np.random.randint(100000, 10000000, n_samples),
        'Duree_Credit':             np.random.randint(6, 84, n_samples),
        'Historique_Defaut':        np.random.choice([0,1], n_samples, p=[0.8,0.2]),  # 20% ont déjà eu un défaut
        'Nombre_Credits_Precedents':np.random.randint(0, 8, n_samples),
        'Charge_Fixe_Mensuelle':    np.random.randint(0, 500000, n_samples),
        'Epargne_Mensuelle':        np.random.randint(0, 500000, n_samples),
        'Defaut':                   np.random.choice([0,1], n_samples, p=[0.75,0.25])  # 25% de défauts de base
    })

    # on force le défaut à 1 pour les cas vraiment risqués :
    # crédit très élevé (> 5 millions) ou charges fixes trop lourdes (> 300 000 FCFA/mois)
    # c'est une règle métier simple qui rend les données plus réalistes
    df['Defaut'] = np.where(
        (df['Montant_Credit'] > 5000000) | (df['Charge_Fixe_Mensuelle'] > 300000),
        1, df['Defaut']
    )
    df['Defaut'] = df['Defaut'].astype(str)  # seaborn a besoin d'une chaîne pour la coloration par catégorie
    return df

# -----------------------------
# Évaluation du modèle
# -----------------------------
def evaluate_model(model, X_test, y_test):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]  # on prend la probabilité de la classe "défaut"

    print("\n===== Classification Report =====")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC Score : {auc:.4f}")

    # la matrice de confusion montre combien de bons et mauvais clients ont été bien classés
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Matrice de Confusion")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.show()

    # la courbe ROC montre à quel point le modèle distingue bien les bons clients des mauvais
    # plus la courbe monte vers le coin haut gauche, meilleur est le modèle
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}", color="#ff7f0e")
    plt.plot([0,1],[0,1], linestyle='--', color='gray')  # ligne de référence : modèle aléatoire
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

# -----------------------------
# Création des figures pour dashboard
# -----------------------------
def create_figures(df, model=None):
    figs = {}  # dictionnaire qui va stocker tous les graphiques avec leur nom comme clé
    numeric_cols = df.select_dtypes(include='number').columns
    num_vars = [c for c in numeric_cols if c != 'Defaut']  # on exclut la cible

    # un histogramme par variable, coloré selon si le client est en défaut ou non
    # le kde=True ajoute une courbe de densité pour mieux voir la forme de la distribution
    for col in num_vars:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.histplot(df, x=col, hue='Defaut', palette=PALETTE, alpha=0.6, kde=True, ax=ax)
        ax.set_title(f"Histogramme de {col} par Defaut")
        figs[f"Histogramme {col}"] = fig

    # les boxplots permettent de voir si une variable sépare bien les deux groupes
    # si les boîtes ne se chevauchent pas, la variable est un bon indicateur de défaut
    for col in num_vars:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.boxplot(x='Defaut', y=col, data=df, palette=PALETTE, ax=ax)
        ax.set_title(f"Boxplot de {col} par Defaut")
        figs[f"Boxplot {col}"] = fig

    # nuages de points pour visualiser les relations entre variables deux à deux
    # chaque point est coloré selon le statut de défaut du client
    for col in num_vars:
        if col not in ['Age','Revenu_Mensuel']:

            # Age vs chaque autre variable
            fig, ax = plt.subplots(figsize=(8,5))
            ax.scatter(
                df['Age'], df[col],
                c=df['Defaut'].map(PALETTE),
                s=50,
                alpha=0.8,       # transparence pour voir les zones denses
                edgecolor='k',
                linewidth=0.3
            )
            ax.set_xlabel('Age')
            ax.set_ylabel(col)
            ax.set_title(f'Age vs {col} coloré par Defaut')
            figs[f"Scatter Age-{col}"] = fig

            # Revenu vs chaque autre variable
            fig, ax = plt.subplots(figsize=(8,5))
            ax.scatter(
                df['Revenu_Mensuel'], df[col],
                c=df['Defaut'].map(PALETTE),
                s=50,
                alpha=0.8,
                edgecolor='k',
                linewidth=0.3
            )
            ax.set_xlabel('Revenu Mensuel')
            ax.set_ylabel(col)
            ax.set_title(f'Revenu vs {col} coloré par Defaut')
            figs[f"Scatter Revenu-{col}"] = fig

    # la heatmap montre les corrélations entre toutes les variables
    # rouge = corrélation positive, bleu = corrélation négative, blanc = pas de lien
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Matrice de corrélation")
    figs["Heatmap corrélation"] = fig

    # si on a un modèle entraîné, on trace sa courbe ROC sur l'ensemble du dataset
    if model is not None:
        X = df[[c for c in df.columns if c != 'Defaut']]
        y = df['Defaut'].astype(int)
        y_pred  = model.predict(X)
        y_proba = model.predict_proba(X)[:,1]
        auc     = roc_auc_score(y, y_proba)

        fig, ax = plt.subplots(figsize=(8,5))
        fpr, tpr, _ = roc_curve(y, y_proba)
        ax.plot(fpr, tpr, label=f"AUC={auc:.3f}", color="#ff7f0e")
        ax.plot([0,1],[0,1], linestyle='--', color='gray')  # référence aléatoire
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        figs["ROC Curve"] = fig

    return figs

# -----------------------------
# Dashboard Tkinter
# -----------------------------
def launch_dashboard(df, model=None):
    figs = create_figures(df, model)  # on génère tous les graphiques avant d'ouvrir la fenêtre

    root = tk.Tk()
    root.title("Dashboard Graphiques Scoring Bancaire")
    root.geometry("1200x800")

    # on divise la fenêtre en deux : liste à gauche, graphique à droite
    list_frame = tk.Frame(root)
    list_frame.pack(side=tk.LEFT, fill=tk.Y)
    fig_frame = tk.Frame(root)
    fig_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    canvas = None  # on garde une référence au graphique affiché pour pouvoir le remplacer

    def show_plot(event):
        nonlocal canvas
        selection = listbox.get(listbox.curselection())
        fig = figs[selection]

        # on retire l'ancien graphique avant d'afficher le nouveau
        if canvas is not None:
            canvas.get_tk_widget().pack_forget()
            canvas.get_tk_widget().place_forget()

        # on intègre le graphique matplotlib dans la fenêtre tkinter
        canvas = FigureCanvasTkAgg(fig, master=fig_frame)
        canvas.draw()
        canvas.get_tk_widget().place(relx=0.5, rely=0.5, anchor="center")

    # liste déroulante avec tous les graphiques disponibles
    listbox = tk.Listbox(list_frame, width=30)
    for key in figs.keys():
        listbox.insert(tk.END, key)
    listbox.pack(fill=tk.Y, expand=True)
    listbox.bind("<<ListboxSelect>>", show_plot)  # un clic sur un nom déclenche l'affichage

    root.mainloop()  # on lance la boucle principale de l'interface
