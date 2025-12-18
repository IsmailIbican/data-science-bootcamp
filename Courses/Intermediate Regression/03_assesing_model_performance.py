import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from itertools import product

# Birden fazla explanatory variable kullanmak fitlerken daha iyi bir sonuç almamıza yardımcı olur 

# Unutma burda R-square ustune çalışcaz

# Büyük değerler iyidir 

# RSE önemli

# RSE küçükse iyidir model

fish = pd.read_csv("fish_data.csv")

mdl_mass_vs_length = ols("mass_g ~ length_cm",
                         data = fish).fit()

mdl_mass_vs_species = ols("mass_g ~ species + 0 ",
                          data = fish).fit()

mdl_mass_vs_both = ols("mass_g ~ length_cm + species + 0",
                       data = fish).fit()

# R-square for mass vs length : 

print(mdl_mass_vs_length.rsquared) # Bunun tutarlılığı da iyi ancak diğeri kadar değil

# R-square for mass vs species : 

print(mdl_mass_vs_species.rsquared) # Bunda ki tahmin çok daha tutarlı diğerine göre

# R-square for mass vs both : 

print(mdl_mass_vs_both.rsquared) # Bunda ise doğruluk oranı çok mu çok daha yüksek

# Adjusted R-squared : 

# Daha fazla explanatory variable r-square değerini arttırmaktadır

# Aşırı derecede explanatory variable'ın olması overfitting'e yol açar

# Bizim elimizdeki fish data set için çok uygun olabilir belki ancak başka bir fish data set için çok kötü olabileceği anlamına geliyor bu 

# Çok uzun bir formülü var bunun : 

""" 
Adjusted r-square = 1 - (1-R^2) * n_obs - 1 / n_obs - n_var - 1
"""

# Bunu kullanmak için : 

#from statsmodels rsquared_adj çağırıyoruz 

# Karşılaştırma kısmı 

print("rsq_length  ", mdl_mass_vs_length.rsquared)

print("rsq_adj_length : ", mdl_mass_vs_length.rsquared_adj)

print("rsq_species : ", mdl_mass_vs_species.rsquared)

print("rsq_adj_species : ", mdl_mass_vs_species.rsquared_adj)

print("rsq_both : ", mdl_mass_vs_both.rsquared)

print("rsq_adj_both : ", mdl_mass_vs_both.rsquared_adj)

# Getting RSE : 

rse_length = np.sqrt(mdl_mass_vs_length.mse_resid) # RSE direkt olarak bulunmaz bunun için MSE'nin square root değerini almamız lazım

print("rse_length : ", rse_length)

rse_species = np.sqrt(mdl_mass_vs_species.mse_resid)

print("rse_species : ", rse_species)

rse_both = np.sqrt(mdl_mass_vs_both.mse_resid)

print("rse_both : ", rse_both)



