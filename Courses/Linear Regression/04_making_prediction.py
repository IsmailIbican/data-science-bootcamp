import numpy as np
import pandas as pd 
import seaborn as sns 
from statsmodels.formula.api import ols 
import matplotlib.pyplot as plt 

# Plotting mass vs length

fishes = pd.read_csv("fish_data.csv")

bream = fishes[fishes["species"] == "Bream"]
print(bream.head())

sns.regplot(x="length_cm", y = "mass_g", data= bream, ci = None)

plt.show()

# Running model ya da fitting model : 

mdl_mass_vs_length = ols("mass_g ~ length_cm", data=bream).fit() # Response solda explanatory variable sağda 

print(mdl_mass_vs_length.params)

# Predict yapmadan önce şunu kendimize sormamız gerekir : 


# Eğerki explanatory variable şu değer de olsaydı response variable'ın değeri ne olurdu ? 

# Yeni bir explanatory variable oluşturmak için bizim tüm explanatory variablelarımızı dataFrame'in içinde tutmamız gerek


# Explanatory data oluşturmak için : 

explanatory_data = pd.DataFrame({"length_cm": np.arange(20,41)}) # Burda biz dictionary kullandık sütun için

# np.arange(20,41) demek 20 den 41 e kadar olan sayılar (41 dahil değil)

# Şimdi predict yapacağız : 

# Bunun için modelimizi çağırıyoruz ve bunun üstünden predict yapıyoruz : 

print(mdl_mass_vs_length.predict(explanatory_data)) # Predict böyle yapılır ve bizim elimizdeki length_cm değerleri ile mass değerlerine predictler yaptık 

# Böyle sadece explanatory variable ile çalışmak pek mantıklı değil o yüzden bizim response variable'a da ihtiyacımız var : 

prediction_data = explanatory_data.assign(
    mass_g = mdl_mass_vs_length.predict(explanatory_data)
)

print(prediction_data)

# Peki ya predicted datalarımızı biz nasıl göstereceğiz : 

fig = plt.figure() # Bomboş bir sayfa açar ve biz bu sayfa üstünde istediğimiz kadar grafiği birleştirebiliriz

sns.regplot(x = "length_cm",
            y = "mass_g",
            data = bream,
            ci = None) # Bu bizim asıl verilerimizin bulundğu regplot ve trend line var linear regression çizgisi 

sns.scatterplot(x = "length_cm",
                y = "mass_g",
                data=prediction_data,
                color = "Red",
                marker = "s")

plt.show()

# Extrapolating : 

# Biz burda belli bir değerler arasında tahmin yürüttük ancak linear regression bu değerler arasında kalmaksızın tahmin yapar biz buna extrapolating deriz

little_bream = pd.DataFrame({"length_cm": [10]})

predict_little_bream = little_bream.assign(
    mass_g = mdl_mass_vs_length(little_bream)
)

print(predict_little_bream)