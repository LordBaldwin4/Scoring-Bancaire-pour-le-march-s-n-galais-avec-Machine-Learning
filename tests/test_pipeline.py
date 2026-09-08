import unittest

from generate_data_senegal import generate_data_senegal
from preprocessing import preprocess


class CreditScoringPipelineTests(unittest.TestCase):
    def test_generation_produit_les_donnees_attendues(self):
        donnees = generate_data_senegal(120)

        self.assertEqual(donnees.shape[0], 120)
        self.assertIn('Defaut', donnees.columns)
        self.assertTrue(donnees['Defaut'].isin([0, 1]).all())

    def test_preprocessing_apprend_le_scaler_sur_train(self):
        donnees = generate_data_senegal(120)

        X_train, X_test, y_train, y_test, colonnes = preprocess(donnees)

        self.assertEqual(X_train.shape[1], len(colonnes))
        self.assertEqual(X_test.shape[0], y_test.shape[0])
        self.assertEqual(X_train.shape[0], y_train.shape[0])


if __name__ == '__main__':
    unittest.main()
