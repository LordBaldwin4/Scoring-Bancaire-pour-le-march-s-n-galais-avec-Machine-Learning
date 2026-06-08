import pandas as pd
import numpy as np
from config import RANDOM_STATE


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_data_senegal(n_samples=2000):
    np.random.seed(RANDOM_STATE)

    # Génération de variables monétaires calibrées sur des ordres de grandeurs réalistes
    revenus = np.random.lognormal(mean=np.log(150000), sigma=0.52, size=n_samples)
    revenus = np.round(np.clip(revenus, 30000, 1500000) / 1000) * 1000

    montants = np.random.lognormal(mean=np.log(1200000), sigma=0.75, size=n_samples)
    montants = np.round(np.clip(montants, 100000, 10000000) / 10000) * 10000

    charges = np.round(np.clip(revenus * np.random.normal(0.18, 0.05, size=n_samples), 20000, 600000) / 1000) * 1000
    epargnes = np.round(np.clip(revenus * np.random.normal(0.12, 0.04, size=n_samples), 0, 350000) / 1000) * 1000

    data = pd.DataFrame({
        'Age':                       np.random.randint(18, 65, size=n_samples),
        'Revenu_Mensuel':            revenus.astype(int),
        'Anciennete_Emploi':         np.clip(np.random.exponential(scale=3.5, size=n_samples).astype(int), 0, 40),
        'Montant_Credit':            montants.astype(int),
        'Duree_Credit':              np.random.randint(6, 85, size=n_samples),
        'Historique_Defaut':         np.random.binomial(1, 0.18, size=n_samples),
        'Nombre_Credits_Precedents': np.clip(np.random.poisson(1.5, size=n_samples), 0, 8),
        'Charge_Fixe_Mensuelle':     charges.astype(int),
        'Epargne_Mensuelle':         epargnes.astype(int)
    })

    # Score logistique probabiliste (sigmoïde + bruit gaussien) pour générer le label
    # Cette approche est plus réaliste qu'un simple if/else et reflète un taux de défaut ~31%
    score = (
        -0.85 * ((data['Revenu_Mensuel'] - 150000) / 100000)
        + 0.45 * ((data['Montant_Credit'] - 1200000) / 2000000)
        + 0.35 * ((data['Charge_Fixe_Mensuelle'] - 100000) / 100000)
        - 0.45 * ((data['Epargne_Mensuelle'] - 20000) / 30000)
        + 0.20 * ((data['Age'] - 35) / 10)
        - 0.30 * ((data['Anciennete_Emploi'] - 3) / 5)
        + 0.30 * ((data['Duree_Credit'] - 24) / 12)
        + 0.80 * data['Historique_Defaut']
        + 0.25 * ((data['Nombre_Credits_Precedents'] - 1) / 2)
    )
    score += np.random.normal(0, 0.75, size=n_samples)

    probabilities = _sigmoid(score - 1.0)
    data['Defaut'] = np.random.binomial(1, probabilities)

    # Ajustement fin pour rester cohérent avec le PAR30 microfinance UEMOA (~31%)
    target_rate = 0.31
    actual_rate = data['Defaut'].mean()
    if abs(actual_rate - target_rate) > 0.01:
        offset = np.log(target_rate / (1 - target_rate)) - np.log(actual_rate / (1 - actual_rate))
        probabilities = _sigmoid(score + offset - 1.0)
        data['Defaut'] = np.random.binomial(1, probabilities)

    return data
