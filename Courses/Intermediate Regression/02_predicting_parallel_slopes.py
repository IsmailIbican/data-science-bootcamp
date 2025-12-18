import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.formula.api import logit
from itertools import product

fish = pd.read_csv("fish_data.csv")

# Aslında aralarında sadece bir tane belirli bir fark var tek explanatory variable ile 

# The prediction workflow : 

# İlk adım : 

# Explanatory variable için değer seçmek : (Single explanatory variable)

expl_data_length = pd.DataFrame({"length_cm" : np.arange(5,61,5)})

print(expl_data_length)

mdl_mass_vs_length = ols("mass_g ~ length_cm",
                         data = fish).fit()
 
# For Multiple Explanatory Variable : 

# Sadece eğer bir tane explanatory variable var ise biz bir tane column hazırlamıştık dataFramemimize 

# Şimdi ise birden fazla column ayarlamalıyız

""" 
Öyle bir dataFrame yapacağiz ki tüm kombinasyonlari tutacak.

Ayni şunun gibi : 


[A,B,C] x [1,2] -> [A1,B1,C1,A2,B2,C2] # Böyle tek tek elle oluşturmak biraz sorun yaratabilir
"""

# Bunun için şu libraryi kullanacağız 

# from itertools import product (Kartezyen Çarpımı)

product(["A","B","C"], [1,2]) # Bize bu fonksiyon kartezyen çarpım olarak döndürecek sonuçları

# Peki bunu datasetimizde nasıl kullanacağız : 

# Explanatory variablelarımız ile bir liste oluşturacağız

length_cm = np.arange(5,61,5) 

species = fish['species'].unique()

# Kartezyen Çarpımı

P = product(length_cm, species)

# Her ikisi için explanatory data frame oluşturma 

expl_data_both = pd.DataFrame(P,
                             columns = ["length_cm",
                                        "species"])


print(expl_data_both)

# Predict mass_g with length_cm only

prediction_data_length = expl_data_length.assign(
    mass_g = mdl_mass_vs_length.predict(expl_data_length)
)

# Kod tamamen aynı model farklı sadece : 

mdl_mass_vs_both = ols("mass_g ~ length_cm + species + 0 ",
                       data = fish).fit()

prediction_data_both = expl_data_both.assign(
    mass_g = mdl_mass_vs_both.predict(expl_data_both)
)

# Taking coeffs : 

coeffs = mdl_mass_vs_both.params

ic_bream, ic_perch, ic_pike, ic_roach, ic_hamsi, ic_palamut, sl = coeffs

# Visualizing the predictions : 

sns.scatterplot(x = "length_cm",
                y = "mass_g",
                hue = "species",
                data = fish)

plt.axline(xy1 = (0,ic_bream), slope = sl, color = "blue")
plt.axline(xy1 = (0,ic_perch), slope = sl, color = "red")
plt.axline(xy1 = (0,ic_pike), slope = sl, color = "orange")
plt.axline(xy1 = (0,ic_roach), slope = sl, color = "yellow")
plt.axline(xy1 = (0,ic_hamsi), slope = sl, color = "purple")
plt.axline(xy1 = (0,ic_palamut), slope = sl, color = "brown")

# Prediction kısmı : 

sns.scatterplot(x = "length_cm",
                y = "mass_g",
                color = "black",
                data = prediction_data_both)

plt.show()

# Mannually calculating predictions : 

coefficients = mdl_mass_vs_length.params

inter, slp = coefficients

explanatory_data = pd.DataFrame({"length_cm" : np.arange(5,61,5)})

prediction_data = explanatory_data.assign(
    mass_g = inter + slp * explanatory_data
)

print(prediction_data)

# Manually calculating predictions on Multiple Explanatory variable : 

# Bunun için np.select() kullanıyoruz

# Argüman olarak list of conditions, list of choices

conditions = [
    expl_data_both["species"] == "Bream",
    expl_data_both["species"] == "Perch",
    expl_data_both["species"] == "Pike", 
    expl_data_both["species"] == "Roach",
    expl_data_both["species"] == "Hamsi",
    expl_data_both["species"] == "Palamut"
]

choices = [ic_bream , ic_perch, ic_pike, ic_roach, ic_hamsi, ic_palamut] # Bunlarda yapacağımız seçimler yani intercept değerlerimiz

# Buradan interceptimizi seçeceğiz spesifik species'e göre

intercept = np.select(conditions, choices)

print(intercept)

# Final step : 

prediction_data_both1 = expl_data_both.assign(
    intercept = np.select(conditions, choices),
    mass_g = intercept + sl + expl_data_both["length_cm"]
)

print(prediction_data_both1)

print(mdl_mass_vs_both.predict(expl_data_both))