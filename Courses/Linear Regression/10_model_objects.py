import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols

# .params ile slope ve intercept değerlerini bulmayı çoktan öğrendik

fish = pd.read_csv("fish_data.csv")

bream = fish[fish["species"] == 'Bream']

mdl_mass_vs_length = ols("mass_g ~ length_cm",
                         data = bream).fit()

print(mdl_mass_vs_length.params)

# Eğer ki orjinal data setimizde biz prediction yapmak istersek şu methotdan yararlanıcaz : 

# .fittedvalues attribute : 

""" 
Jargon : is jargon for predictions on the original dataset used to create the model
"""

# Aslında explanatory_variable ile yaptığımızın kısa yoludur bu 

print(mdl_mass_vs_length.fittedvalues) # NaN değerlerini tahmin edemez onun için genelde machine learning tabanlı algoritmalar kullanılır

# Ya da bunu başka bir şekilde de yapabiliriz : 

explanatory_data = bream["length_cm"]

print(mdl_mass_vs_length.predict(explanatory_data))

# .resid ne işe yarar : 

# .resid her bir tahmin ne kadar sapma yapmış onu verir 

# Aslında bir nevi doğruluk payını söyler bize.

""" 
Actual response values - predicted response values (Yani sapma değeri)
"""

print(mdl_mass_vs_length.resid)

# Ya da şöyle yapılabilir : 

print(bream['mass_g'] - mdl_mass_vs_length.fittedvalues) #Bream balığının mass_g değerlerinden tüm tahmin edilen mass_g değerlerini çıkart

