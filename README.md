# 🏦 Projet Scoring Bancaire – Data Science

> Simulation complète d'un **système de scoring bancaire** basé sur le Machine Learning.  
> Génération de données synthétiques (contexte sénégalais), prétraitement, entraînement de modèles et visualisation interactive.

---

## ⚙️ Fonctionnement au lancement

Quand tu exécutes le projet, voici ce qui se passe **dans l'ordre** :

1. **Un graphique s'ouvre** (matrice de confusion + courbe ROC) → **ferme-le** pour continuer
2. **Un dashboard interactif s'ouvre** (interface Tkinter) → tu peux y sélectionner et afficher tous les graphiques disponibles

---

## 📁 Structure du projet

```
scoring-bancaire/
│
├── config.py                           # Paramètres globaux (seed, taille test, etc.)
├── generate_data_senegal.py            # Génération du dataset synthétique
├── preprocessing.py                    # Préparation des données (split, scaling)
├── model_logistic.py                   # Modèle de régression logistique
├── model_random_forest.py              # Modèle Random Forest avec GridSearch
├── eda.py                              # Analyse exploratoire (EDA)
├── evaluation_scoring_dashboard.py     # Évaluation + dashboard Tkinter  ← POINT D'ENTRÉE
├── test_plot.py                        # Test rapide d'affichage matplotlib
├── requirements.txt                    # Dépendances du projet
└── README.md
```

---

## 🚀 Installation et lancement

### Étape 1 — Cloner le projet

```bash
git clone <url-du-repo>
cd scoring-bancaire
```

### Étape 2 — Créer un environnement virtuel (venv)

> ⚠️ **Recommandé** pour isoler les dépendances du projet et éviter les conflits avec ton système.

```bash
# Créer le venv
python -m venv venv

# Activer le venv — Windows :
venv\Scripts\activate

# Activer le venv — Mac/Linux :
source venv/bin/activate
```

> Une fois activé, tu verras `(venv)` apparaître dans ton terminal.

### Étape 3 — Installer les dépendances

```bash
pip install -r requirements.txt
```

### Étape 4 — Lancer le projet

```bash
python evaluation_scoring_dashboard.py
```

---

## 📊 Ce que tu verras au lancement

### 1️⃣ Graphiques d'évaluation (fenêtre matplotlib)

À l'ouverture, deux graphiques s'affichent automatiquement :
- **Matrice de confusion** du modèle Random Forest
- **Courbe ROC** avec le score AUC

> ➡️ **Ferme cette fenêtre** pour accéder au dashboard interactif.

### 2️⃣ Dashboard interactif (fenêtre Tkinter)

Une interface s'ouvre avec une **liste de graphiques** à gauche.  
Clique sur un nom pour afficher le graphique correspondant à droite :

| Graphique | Description |
|---|---|
| Histogramme [variable] | Distribution de chaque variable selon le défaut |
| Boxplot [variable] | Comparaison de chaque variable selon le défaut |
| Heatmap corrélation | Corrélation entre toutes les variables |
| ROC Curve | Performance globale du modèle (AUC) |

---

## 🧠 Modèles utilisés

| Modèle | Fichier | Description |
|---|---|---|
| Régression Logistique | `model_logistic.py` | Modèle de référence, interprétable et rapide |
| Random Forest | `model_random_forest.py` | Modèle ensembliste avec optimisation GridSearch (cv=5) |

---

## 🔧 Désactiver le venv

Une fois que tu as fini, désactive l'environnement virtuel :

```bash
deactivate
```

---

## 📦 Dépendances principales

| Librairie | Utilisation |
|---|---|
| `pandas` / `numpy` | Manipulation des données |
| `scikit-learn` | Modèles ML, évaluation, prétraitement |
| `matplotlib` / `seaborn` | Visualisation des graphiques |
| `tkinter` | Dashboard interactif (inclus nativement avec Python) |

---

## 🌍 Contexte des données

Le dataset est **synthétique**, simulé avec des caractéristiques socio-économiques adaptées au contexte sénégalais : revenus en **FCFA**, montants de crédit, charges fixes mensuelles, épargne, historique de défaut, etc.

---

## 👤 Auteur

> Projet réalisé dans le cadre d'une formation en **Data Science / Machine Learning**.

