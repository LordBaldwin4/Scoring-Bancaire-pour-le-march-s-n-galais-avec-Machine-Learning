# Projet Scoring Bancaire – Data Science

Ce projet permet de **simuler un scoring bancaire** à l'aide de modèles de machine learning.  
Il inclut la génération d’un dataset synthétique, le prétraitement, l’entraînement de modèles et la visualisation complète des résultats (rapports, matrices, courbes ROC et graphiques exploratoires).

---

## Contenu du projet

- **generate_data_senegal.py** : génération du dataset synthétique pour le scoring bancaire.  
- **preprocessing.py** : fonctions de préparation des données (split train/test, sélection des features).  
- **model_logistic.py** : entraînement d’un modèle de régression logistique.  
- **model_random_forest.py** : entraînement d’un modèle Random Forest.  
- **evaluation_scoring_dashboard.py** : fonctions pour l’évaluation, création des figures et dashboard Tkinter.  
- **Notebook.ipynb** : version interactive du projet (recommandée) qui affiche tous les graphiques directement dans Jupyter.  

---

## Prérequis

- Python ≥ 3.9  
- Librairies Python :  
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
