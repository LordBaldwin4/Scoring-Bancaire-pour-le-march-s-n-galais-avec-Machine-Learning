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
RANDOM_STATE = 42
sns.set_theme(style="darkgrid")
PALETTE = {'0': "#1f77b4", '1': "#ff7f0e"}  # Bleu = pas défaut, Orange = défaut

# -----------------------------
# Génération dataset
# -----------------------------
def generate_data_senegal(n_samples=3000):
    np.random.seed(RANDOM_STATE)
    df = pd.DataFrame({
        'Age': np.random.randint(18, 65, n_samples),
        'Revenu_Mensuel': np.random.randint(50000, 1500000, n_samples),
        'Anciennete_Emploi': np.random.randint(0, 35, n_samples),
        'Montant_Credit': np.random.randint(100000, 10000000, n_samples),
        'Duree_Credit': np.random.randint(6, 84, n_samples),
        'Historique_Defaut': np.random.choice([0,1], n_samples, p=[0.8,0.2]),
        'Nombre_Credits_Precedents': np.random.randint(0, 8, n_samples),
        'Charge_Fixe_Mensuelle': np.random.randint(0, 500000, n_samples),
        'Epargne_Mensuelle': np.random.randint(0, 500000, n_samples),
        'Defaut': np.random.choice([0,1], n_samples, p=[0.75,0.25])
    })
    df['Defaut'] = np.where(
        (df['Montant_Credit'] > 5000000) | (df['Charge_Fixe_Mensuelle'] > 300000),
        1, df['Defaut']
    )
    df['Defaut'] = df['Defaut'].astype(str)  # Pour Seaborn
    return df

# -----------------------------
# Évaluation du modèle
# -----------------------------
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    print("\n===== Classification Report =====")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC Score : {auc:.4f}")

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Matrice de Confusion")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}", color="#ff7f0e")
    plt.plot([0,1],[0,1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

# -----------------------------
# Création des figures pour dashboard
# -----------------------------
def create_figures(df, model=None):
    figs = {}
    numeric_cols = df.select_dtypes(include='number').columns
    num_vars = [c for c in numeric_cols if c != 'Defaut']

    # Histogrammes
    for col in num_vars:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.histplot(df, x=col, hue='Defaut', palette=PALETTE, alpha=0.6, kde=True, ax=ax)
        ax.set_title(f"Histogramme de {col} par Defaut")
        figs[f"Histogramme {col}"] = fig

    # Boxplots
    for col in num_vars:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.boxplot(x='Defaut', y=col, data=df, palette=PALETTE, ax=ax)
        ax.set_title(f"Boxplot de {col} par Defaut")
        figs[f"Boxplot {col}"] = fig

    # Scatter Age/Revenu vs autres
    for col in num_vars:
        if col not in ['Age','Revenu_Mensuel']:
            # Age vs col
            fig, ax = plt.subplots(figsize=(8,5))
            ax.scatter(
                df['Age'], df[col], 
                c=df['Defaut'].map(PALETTE), 
                s=50,
                alpha=0.8,
                edgecolor='k',
                linewidth=0.3
            )
            ax.set_xlabel('Age')
            ax.set_ylabel(col)
            ax.set_title(f'Age vs {col} coloré par Defaut')
            figs[f"Scatter Age-{col}"] = fig

            # Revenu vs col
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

    # Heatmap
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Matrice de corrélation")
    figs["Heatmap corrélation"] = fig

    # ROC si modèle fourni
    if model is not None:
        X = df[[c for c in df.columns if c != 'Defaut']]
        y = df['Defaut'].astype(int)
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:,1]
        auc = roc_auc_score(y, y_proba)

        fig, ax = plt.subplots(figsize=(8,5))
        fpr, tpr, _ = roc_curve(y, y_proba)
        ax.plot(fpr, tpr, label=f"AUC={auc:.3f}", color="#ff7f0e")
        ax.plot([0,1],[0,1], linestyle='--', color='gray')
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
    figs = create_figures(df, model)
    root = tk.Tk()
    root.title("Dashboard Graphiques Scoring Bancaire")
    root.geometry("1200x800")

    list_frame = tk.Frame(root)
    list_frame.pack(side=tk.LEFT, fill=tk.Y)
    fig_frame = tk.Frame(root)
    fig_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    canvas = None

    def show_plot(event):
        nonlocal canvas
        selection = listbox.get(listbox.curselection())
        fig = figs[selection]

        if canvas is not None:
            canvas.get_tk_widget().pack_forget()
            canvas.get_tk_widget().place_forget()

        canvas = FigureCanvasTkAgg(fig, master=fig_frame)
        canvas.draw()
        canvas.get_tk_widget().place(relx=0.5, rely=0.5, anchor="center")

    listbox = tk.Listbox(list_frame, width=30)
    for key in figs.keys():
        listbox.insert(tk.END, key)
    listbox.pack(fill=tk.Y, expand=True)
    listbox.bind("<<ListboxSelect>>", show_plot)

    root.mainloop()
