<div align="center">

# 🏦 Scoring Bancaire — Marché Sénégalais
### Système de prédiction de défaut de crédit par Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.0-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Licence](https://img.shields.io/badge/Licence-MIT-22C55E?style=for-the-badge)
![Statut](https://img.shields.io/badge/Statut-Terminé-22C55E?style=for-the-badge)

*Simulation complète d'un pipeline de scoring bancaire adapté au contexte économique sénégalais — de la génération des données à l'évaluation des modèles.*

</div>

---

## 📋 Présentation

Le **scoring bancaire** est une méthode utilisée par les institutions financières pour évaluer la probabilité qu'un client fasse défaut sur un crédit. Ce projet simule un pipeline complet de scoring adapté au marché sénégalais :

- 📊 Génération d'un dataset synthétique **réaliste** (2 000 clients, chiffres en FCFA)
- 🔧 Preprocessing et standardisation des données
- 🤖 Entraînement de deux modèles ML avec optimisation des hyperparamètres
- 📈 Évaluation complète avec AUC-ROC, matrices de confusion, courbes ROC
- 🖥️ Tableau de bord interactif Tkinter

---

## 📊 Résultats des modèles

> Les résultats ci-dessous sont obtenus sur **400 observations de test** (20% du dataset).  
> Le taux de défaut dans les données est de **31.6%**, calibré sur les données BCEAO/UEMOA.

| Métrique | Random Forest | Régression Logistique |
|---|:---:|:---:|
| **AUC-ROC** | 0.743 | **0.765** |
| Accuracy | 69% | **71%** |
| Précision — classe Défaut | 50% | 53% |
| Rappel — classe Défaut | 67% | **73%** |
| F1-score — classe Défaut | 0.57 | **0.61** |
| Faux positifs | 84 | **81** |
| Faux négatifs | 41 | **34** |

**Lecture des résultats :**
- La **Régression Logistique** obtient le meilleur AUC (0.796) et détecte mieux les défauts (rappel 71%) — idéale si minimiser les faux négatifs est prioritaire (ne pas rater un mauvais payeur).
- Le **Random Forest** génère moins de faux positifs (63 vs 86) — idéal si minimiser les refus injustifiés de bons clients est prioritaire.
- Le choix entre les deux dépend du **coût métier** associé à chaque type d'erreur.

---

## 🖼️ Visualisations

### Matrices de Confusion

![Matrices de Confusion](confusion_matrix.PNG)

### Courbes ROC

![Courbes ROC](roc_curve.PNG)

### Matrice de Corrélation

![Heatmap Corrélation](heatmap.PNG)

### Tableau de Bord — Aperçu du Dataset

![Dashboard](dashboard.PNG)

---

## 🗂️ Structure du projet

```
scoring-bancaire-senegal/
│
├── config.py                    # Paramètres globaux : RANDOM_STATE, TEST_SIZE, N_SAMPLES
├── generate_data_senegal.py     # Génération du dataset synthétique réaliste (FCFA)
├── preprocessing.py             # Split train/test stratifié + StandardScaler
├── model_logistic.py            # Régression Logistique baseline + évaluation complète
├── model_random_forest.py       # Random Forest + GridSearchCV (cv=5, scoring=AUC)
├── eda.py                       # Analyse exploratoire des données
├── evaluation_scoring_dashboard.py  # Pipeline complet + tableau de bord Tkinter
│
├── confusion_matrix.PNG         # Matrices de confusion RF et LR
├── roc_curve.PNG                # Courbes ROC comparées
├── heatmap.PNG                  # Matrice de corrélation des features
├── dashboard.PNG                # Aperçu distributions et feature importances
│
└── README.md
```

---

## 🧮 Données synthétiques

Le dataset est généré pour refléter le tissu socio-économique sénégalais.  
Le label `Defaut` est produit via un **score logistique probabiliste avec bruit** — jamais par des règles déterministes, ce qui garantit des résultats ML réalistes (un AUC = 1.0 serait le signe d'un *data leakage*).

| Feature | Type | Distribution | Calibration |
|---|---|---|---|
| `Age` | int | Uniforme [21, 62] | Population active sénégalaise |
| `Revenu_Mensuel` | int (FCFA) | Log-normale, médiane ~150 000 | Enquête Emploi ANSD 2021 |
| `Anciennete_Emploi` | int (années) | Corrélée à l'âge | Logique métier |
| `Montant_Credit` | int (FCFA) | Log-normale, [100k – 15M] | Grille microfinance BCEAO |
| `Duree_Credit` | int (mois) | Catégorielle [6 – 84] | Pratiques bancaires UEMOA |
| `Historique_Defaut` | binaire | 18% d'incidents | Données BCEAO |
| `Charge_Fixe_Mensuelle` | int (FCFA) | Beta × Revenu | Ratio charges/revenu |
| `Epargne_Mensuelle` | int (FCFA) | Corrélée au revenu | Logique métier |
| `Nombre_Credits_Precedents` | int | Catégorielle [0 – 7] | Profils microfinance |
| **`Defaut`** | **binaire** | **31.6% de défauts** | **PAR30 microfinance UEMOA** |

---

## ⚙️ Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/LordBaldwin4/Scoring-Bancaire-pour-le-march-s-n-galais-avec-Machine-Learning.git
cd Scoring-Bancaire-pour-le-march-s-n-galais-avec-Machine-Learning

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Mac / Linux
# venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer le pipeline complet
python evaluation_scoring_dashboard.py
```

---

## 🔬 Modèles

| Modèle | Fichier | Particularités |
|---|---|---|
| **Régression Logistique** | `model_logistic.py` | Baseline interprétable, `C=0.1`, `class_weight='balanced'`, `max_iter=1000` |
| **Random Forest** | `model_random_forest.py` | GridSearchCV 48 combinaisons, `cv=5`, `scoring='roc_auc'`, `class_weight='balanced'` |

---

## 📦 Dépendances

| Librairie | Version | Usage |
|---|---|---|
| `pandas` | ≥ 2.0 | Manipulation des données |
| `numpy` | ≥ 1.24 | Calculs numériques |
| `scikit-learn` | ≥ 1.3 | Modèles ML, preprocessing, évaluation |
| `matplotlib` | ≥ 3.7 | Graphiques |
| `seaborn` | ≥ 0.12 | Heatmaps et visualisations statistiques |
| `tkinter` | stdlib | Tableau de bord interactif |

---

## 👤 Auteur

**Bassirou Ousmane Ba**  
Étudiant en Génie Logiciel et Systèmes d'Information — ESP/UCAD, Dakar, Sénégal  

---

## 📄 Licence

Ce projet est distribué sous licence **MIT**.
