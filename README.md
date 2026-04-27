# Banques — Faire un système de crédit avec l'apprentissage automatique

![Python](https://img.shields.io/badge/Python-3.8-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn&logoColor=white)
![Licence](https://img.shields.io/badge/Licence-MIT-green)
![Statut](https://img.shields.io/badge/Statut-Terminé-brightgreen)

Simulation complète d'un système de scoring bancaire qui utilise le Machine Learning.
Créer des données artificielles qui ressemblent à celles du Sénégal, les préparer, faire fonctionner des modèles dessus et voir les résultats de manière interactive.

---

## Résultats des modèles

| Métrique | Random Forest | Régression Logistique |
|---|---|---|
| AUC | 1.000 | — |
| Précision | Parfaite | — |
| Faux positifs | 0 | — |
| Faux négatifs | 0 | — |

Le modèle Random Forest marche super bien avec les données qu'on lui a fabriquées.

---

## Illustrations

### Matrice de Confusion — Forêt Aléatoire

![Matrice de confusion](confusion_matrix.PNG)

Il n'y a pas eu d'erreurs, on a eu 211 cas où la réponse était non et elle aurait dû l'être, et 789 cas où la réponse était oui et elle aurait dû l'être.

---

### Courbe ROC — L'AUC est de 1

![Courbe ROC](roc_curve.PNG)

Quand la courbe ROC monte jusqu'en haut à gauche, ça veut dire que le système est super bon pour distinguer les bons clients des mauvais.

---

## Tableau de bord interactif

![Dashboard Tkinter](dashboard.PNG)

Une interface Tkinter propose plus de vingt types de graphiques, comme des histogrammes, des diagrammes en boîte, des cartes de chaleur pour les corrélations, ou même des courbes ROC.

### Matrice de Corrélation

![Heatmap Corrélation](heatmap.PNG)

Les variables sont à peu près indépendantes, avec des corrélations proches de zéro, sauf quelques petites liaisons entre Montant Crédit et Défaut (0.46) et la Charge Fixe Mensuelle divisée par Défaut (0.33).

---

## Comment ça marche quand on le lance

Quand tu fais le projet, voici dans quel ordre les choses se passent :

1. Un graphique va apparaître (c'est une matrice de confusion avec une courbe ROC), tu peux le fermer pour avancer.
2. Un tableau de bord interactif s'ouvre, c'est comme une interface Tkinter. Il permet de choisir et de montrer tous les graphiques qu'on peut voir.

---

## Structure du projet
```
scoring-bancaire/
├── config.py                  # Quelques réglages de base (comme le point de départ, la taille des données de test, etc.)
├── generate_data_senegal.py   # Construire notre dataset de toutes pièces
├── preprocessing.py           # Préparation des données (séparation, mise à l'échelle)
├── model_logistique.py        # Modèle de régression logistique
├── model_random_forest.py     # Modèle Random Forest et recherche par quadrillage
├── eda.py                     # Exploration des données
├── evaluation_dashboard.py    # Évaluation + tableau de bord Tkinter ← point de départ du programme
├── test_plot.py               # Un petit test pour voir comment matplotlib affiche les choses
├── requirements.txt           # Les dépendances du projet
└── README.md
```

---

## Installation et démarrage

### Étape 1 — Cloner le projet
```bash
git clone https://github.com/bassirou-ousmane-ba/scoring-bancaire.git
cd scoring-bancaire
```

### Étape 2 — Construire une machine virtuelle

Recommandé pour pouvoir séparer les dépendances et ne pas avoir de soucis.
```bash
python -m venv venv
```

Activer Windows :
```bash
venv\Scripts\activate
```

Activer Mac/Linux :
```bash
source venv/bin/activate
```

### Étape 3 — Installer les dépendances
```bash
pip install -r requirements.txt
```

### Étape 4 — On démarre le projet
```bash
python evaluation_dashboard.py
```

---

## Modèles employés

| Modèle | Fichier | Description |
|---|---|---|
| Régression Logistique | model_logistique.py | C'est un modèle qui est rapide, on peut le comprendre facilement, et il sert de référence. |
| Forêt Aléatoire | model_random_forest.py | Modèle d'ensemble GridSearch (cv=5) |

---

## Graphiques dans le tableau de bord

| Catégorie | Graphiques |
|---|---|
| Histogrammes | Age, Revenu Mensuel, Ancienneté Emploi, Montant Credit, Durée Credit, Historique Défaut, Nombre Credits Précédents, Charge Fixe Mensuelle, Épargne Mensuelle |
| Boîtes à moustaches | Les mêmes variables comparées en fonction du défaut |
| Corrélation | Une carte de chaleur qui montre comment toutes les variables sont liées entre elles |
| Évaluation | Courbe ROC avec score AUC |

---

## Contexte des données

La base de données est synthétique, et on l'a faite en reproduisant des situations socio-économiques comme on en trouve au Sénégal.

- Revenus en francs CFA
- Montants des crédits dans le coin
- Les frais mensuels fixes
- Économies mensuelles
- Historique de défauts

---

## Principales dépendances

| Librairie | Utilisation |
|---|---|
| pandas, numpy | Traitement des données |
| scikit-learn | Modèles pour l'apprentissage automatique, outils pour les évaluer et pour préparer les données |
| matplotlib, seaborn | Afficher les graphiques |
| tkinter | Un tableau de bord interactif inclus dans Python |

---

## Désactiver l'environnement virtuel
```bash
deactivate
```

---

## Licence

Ce projet utilise la licence MIT.

---

## Auteur

Bassirou Ousmane Ba
