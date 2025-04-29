#%%
import pandas as pd

# Carica il file
df = pd.read_excel(r'C:\Users\glmar\OneDrive\Bemacs Bocconi\Secondo semestre\Machine Learning\ML project\ML project data\train.xlsx') 

# Numero di righe e colonne
features = df.shape[1]
n_features = features - 1
print(f'Number of features: {n_features}')

# Statistiche su value_eur
print(df['value_eur'].describe())
#%%
# Istogramma
import matplotlib.pyplot as plt
plt.hist(df['value_eur'], bins=50)
plt.xlabel('Valore (€)')
plt.ylabel('Numero di giocatori')
plt.title('Distribuzione del valore di mercato')
plt.xlim(0, 80000000)  # cambia il limite come vuoi (es. 5 milioni di euro)
plt.show()

