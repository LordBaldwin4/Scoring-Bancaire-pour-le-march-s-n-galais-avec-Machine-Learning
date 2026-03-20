import pandas as pd
import numpy as np
from config import RANDOM_STATE  # assure-toi que config.py contient RANDOM_STATE

def generate_data_senegal(n_samples=2000):
    np.random.seed(RANDOM_STATE)  # graine fixe pour reproduire exactement les mêmes données à chaque fois

    # on fabrique 2000 clients fictifs mais réalistes, inspirés du contexte économique sénégalais
    # toutes les valeurs monétaires sont en FCFA
    data = pd.DataFrame({
        'Age':                       np.random.randint(18, 65, n_samples),           # entre 18 et 65 ans, âge actif
        'Revenu_Mensuel':            np.random.randint(50000, 1500000, n_samples),   # FCFA — du petit salaire au revenu confortable
        'Anciennete_Emploi':         np.random.randint(0, 35, n_samples),            # en années, de débutant à vétéran
        'Montant_Credit':            np.random.randint(100000, 10000000, n_samples), # FCFA — du petit crédit au gros prêt
        'Duree_Credit':              np.random.randint(6, 84, n_samples),            # en mois, de 6 mois à 7 ans
        'Historique_Defaut':         np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),  # 20% ont déjà eu un défaut de paiement
        'Nombre_Credits_Precedents': np.random.randint(0, 8, n_samples),             # combien de crédits le client a déjà eu
        'Charge_Fixe_Mensuelle':     np.random.randint(0, 500000, n_samples),        # FCFA — loyer, factures, remboursements en cours
        'Epargne_Mensuelle':         np.random.randint(0, 500000, n_samples),        # FCFA — ce que le client met de côté chaque mois
        'Defaut':                    np.random.choice([0, 1], n_samples, p=[0.75, 0.25])  # 25% de défauts de base dans la population
    })

    # on applique une règle métier simple pour rendre les données plus réalistes :
    # si le crédit dépasse 5 millions ou si les charges fixes dépassent 300 000 FCFA/mois,
    # on force le défaut à 1 — ce sont des profils clairement à risque
    data['Defaut'] = np.where(
        (data['Montant_Credit'] > 5000000) | (data['Charge_Fixe_Mensuelle'] > 300000),
        1,
        data['Defaut']  # sinon on garde la valeur tirée au sort plus haut
    )

    return data
