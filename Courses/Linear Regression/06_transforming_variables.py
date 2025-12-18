import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from statsmodels.formula.api import ols

# Bazen explanatory variable ile response variable arasındaki ilişki düz bir çizgi halinde olmayabilir

# Regression model oluşturmak için bizim bu verilerimizi transform etmemiz gerekebilir

fishes = pd.read_csv("fish_data.csv")

perch = fishes[fishes["species"] == "Perch"]

print(perch.head())

# As you can see it is no a linear relationship

sns.regplot(x = "length_cm", y= "mass_g", data = fishes, ci = None)

plt.show()

# Let's try to cube the length when we seed the perch's image we can predict that when it's growth it might be taller bigger rounder longer etc...

# Bunu yapmak için bizim yeni bir tane column oluşturmamız lazım : 

perch["length_cm_cubed"] = perch["length_cm"] ** 3

sns.regplot(x = "length_cm_cubed", y="mass_g", data = perch, ci = None)

plt.show()

# Modeling mass vs length cubed : 

mdl_perch = ols("mass_g ~ length_cm_cubed",
                               data = perch).fit()

print(mdl_perch.params)

# Predicting mass vs length cubed : 

explanatory_data = pd.DataFrame({"length_cm_cubed" : np.arange(10,41,5) ** 3,
                                 "length_cm" : np.arange(10,41,5)}) # Biz burada transforme edilmemiş bir referans sütunu olarak length_cm'yi de tutuyoruz ki arasındaki farkı görmek için şimdi iki değer için de biz predict yapcağız değişimleri nasıl olacak onlara bakacağız

print(mdl_perch.predict(explanatory_data))

# Or : 

prediction_data = explanatory_data.assign(
    mass_g = mdl_perch.predict(explanatory_data)
)

print(prediction_data)

# Plotting mass vs length cubed : 

fig = plt.figure()

sns.regplot(x = "length_cm_cubed", y= "mass_g",
            data = perch,
            ci = None)

sns.scatterplot(x = "length_cm_cubed", y = "mass_g",
                data = prediction_data,
                color = "red",
                marker = "s")

plt.show()

facebook = pd.read_csv("facebook_data.csv")

sns.regplot(x = "spent_usd", y = "n_impression",
            data = facebook,
            ci = None)

plt.show()

# Bu grafikte normalde bizim değerlerimiz orijine yakın bir değerde toplanmışlar genel olarak bi sıkışıklık var her iki değer de onun için biz burda square root yontemini  kullanıcaz : 

facebook["sqrt_spent_usd"] = np.sqrt(facebook["spent_usd"])

facebook["sqrt_n_impression"] = np.sqrt(facebook["n_impression"])

sns.regplot(x = "sqrt_spent_usd", y = "sqrt_n_impression",
            data = facebook,
            ci = None)

plt.show()

# Square root transformation daha çok right-skewed dataları düzenlemek için kullanılır

# Modeling and predicting : 

mdl_ad = ols("sqrt_n_impression ~ sqrt_spent_usd",
             data = facebook).fit()

explanatory_data1 = pd.DataFrame({"sqrt_spent_usd": np.arange(0,601,100),
                                 "spent_usd" : np.arange(0,601,100)})

print(mdl_ad.predict(explanatory_data1))

prediction_data1 = explanatory_data1.assign(
    sqrt_n_impression = mdl_ad.predict(explanatory_data1) ** 2
)

print(prediction_data1)

fig = plt.figure()

sns.regplot(x = "sqrt_spent_usd",
            y = "sqrt_n_impression",
            data = facebook,
            ci = None)

sns.scatterplot(x = "sqrt_spent_usd",
                y = "sqrt_n_impression",
                data = prediction_data1,
                color="red",
                marker= "s")

plt.show()