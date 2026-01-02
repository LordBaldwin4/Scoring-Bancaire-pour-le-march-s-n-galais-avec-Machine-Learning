import pandas as pd
import numpy as np
from config import RANDOM_STATE  # assure-toi que config.py contient RANDOM_STATE

def generate_data_senegal(n_samples=2000):
    np.random.seed(RANDOM_STATE)

    data = pd.DataFrame({
        'Age': np.random.randint(18, 65, n_samples),
        'Revenu_Mensuel': np.random.randint(50000, 1500000, n_samples),  # FCFA
        'Anciennete_Emploi': np.random.randint(0, 35, n_samples),
        'Montant_Credit': np.random.randint(100000, 10000000, n_samples),  # FCFA
        'Duree_Credit': np.random.randint(6, 84, n_samples),  # mois
        'Historique_Defaut': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Nombre_Credits_Precedents': np.random.randint(0, 8, n_samples),
        'Charge_Fixe_Mensuelle': np.random.randint(0, 500000, n_samples),   # FCFA
        'Epargne_Mensuelle': np.random.randint(0, 500000, n_samples),       # FCFA
        'Defaut': np.random.choice([0, 1], n_samples, p=[0.75, 0.25])
    })

   
    data['Defaut'] = np.where(
        (data['Montant_Credit'] > 5000000) | (data['Charge_Fixe_Mensuelle'] > 300000),
        1,
        data['Defaut']
    )

    return data
