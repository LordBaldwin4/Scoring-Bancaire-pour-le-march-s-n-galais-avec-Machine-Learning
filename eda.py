import matplotlib.pyplot as plt

def plot_default_distribution(data):
    data['Defaut'].value_counts().plot(kind='bar')
    plt.title("Distribution du défaut de crédit")
    plt.xlabel("Défaut")
    plt.ylabel("Nombre de clients")
    plt.show()
