import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from statsmodels.formula.api import ols 
from itertools import product

fish = pd.read_csv("fish_data.csv")

print(fish["species"].unique())

# Parallel Slope ile yaptığımız da her bir species için farklı bi slope yapıp onları modelliyoruz

# İlk başta biz bunları 4 tane farklı datasete ayırıyoruz : 

bream = fish[fish["species"] == "Bream"]
perch = fish[fish["species"] == "Perch"]
pike = fish[fish["species"] == "Pike"]
roach = fish[fish["species"] == "Roach"]
hamsi = fish[fish["species"] == "Hamsi"]
palamut = fish[fish["species"] == "Palamut"]

# Şimdi elimiz de birbirinden fakrlı 6 tane data setimiz oldu 

mdl_bream = ols("mass_g ~ length_cm",
                data = bream).fit()

print("Params of bream : ",mdl_bream.params)

mdl_perch = ols("mass_g ~ length_cm",
                data = perch).fit()

print("Params of perch : ",mdl_perch.params)

mdl_pike = ols("mass_g ~ length_cm",
                data = pike).fit()

print("Params of pike : ",mdl_pike.params)

mdl_roach= ols("mass_g ~ length_cm",
                data = roach).fit()

print("Params of roach : ",mdl_roach.params)

mdl_hamsi = ols("mass_g ~ length_cm",
                data = hamsi).fit()

print("Params of hamsi : ",mdl_hamsi.params)

mdl_palamut = ols("mass_g ~ length_cm",
                data = palamut).fit()

print("Params of palamut : ",mdl_palamut.params)

# Klasik explanatory variable oluşturmamız gerekli ve her bir farklı data setimizin bir tane explanatory variableı var

explanatory_data = pd.DataFrame({"length_cm" : np.arange(4, 34, 0.5)})

print(explanatory_data)

# Making predictions : 

prediction_data_bream = explanatory_data.assign(
    mass_g = mdl_bream.predict(explanatory_data),
    species = "Bream"
)

prediction_data_perch = explanatory_data.assign(
    mass_g = mdl_perch.predict(explanatory_data),
    species = "Perch"
)

prediction_data_pike = explanatory_data.assign(
    mass_g = mdl_pike.predict(explanatory_data),
    species = "Pike"
)

prediction_data_roach = explanatory_data.assign(
    mass_g = mdl_roach.predict(explanatory_data),
    species = "Roach"
)

prediction_data_hamsi = explanatory_data.assign(
    mass_g = mdl_hamsi.predict(explanatory_data),
    species = "Hamsi"
)

prediction_data_palamut = explanatory_data.assign(
    mass_g = mdl_palamut.predict(explanatory_data),
    species = "Palamut"
)

# Bütün bu ayrı predictionları biz tek bir dataFrame üstünde toplamak istersek concat fonksiyonunu kullanıyoruz

# Concatenating Predictions : 

prediction_data = pd.concat([prediction_data_bream,
                             prediction_data_perch,
                             prediction_data_pike,
                             prediction_data_roach,
                             prediction_data_hamsi,
                             prediction_data_palamut])

# Visualizing predictions : 

# Subset of dataFrames olduğu için regplot kullanamayız biz burada

sns.lmplot(x = "length_cm",
           y = "mass_g",
           data =fish,
           hue = "species",
           ci=None)


# Burda parallel slopelar yok gördüğümüz üzere her birinin ayrı slope değeri ve predictionları var

# Adding in your predictions 

sns.scatterplot(x="length_cm",
                y = "mass_g",
                data=prediction_data,
                hue="species",
                legend=False,
                color="black")

plt.show()

# Coefficient of determination : 

mdl_fish = ols("mass_g ~ length_cm + species + 0 ",
               data = fish).fit()

print(mdl_fish.rsquared_adj)

# RSE of Model : 

print(np.sqrt(mdl_fish.mse_resid))