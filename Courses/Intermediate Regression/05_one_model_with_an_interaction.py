import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from statsmodels.formula.api import ols 
from itertools import product

fish = pd.read_csv("fish_data.csv")

# What is an interaction : 

# In the fish dataset ; 

# Her bir farklı balık türünün farklı uzunluk ve ağırlık oranları vardır

# Bunu istatistiksel olarak söylemek gerekirse de uzunluğun tahmin edilen ağırlık üzerindeki etkisi türden türe değişmektedir

# Diğer bir yandan bir explanatory variable diğer bir explanatory variable değerine göre tahmin edilen response variable'a göre değişmektedir

# Birden fazla explanatory variable'ın kullanımını gördük biz 

mdl_mass_vs_both = ols("mass_g ~ length_cm + species + 0",
                       data = fish).fit()

# Örtülü interaction : (Implicit)

# Aslında burada hangi variableları kullandığımızı göstermemizin bir önemi yok çünkü adı üstünde Implicit

mdl_mass_vs_both_implicit = ols("mass_g ~ length_cm * species",
                                data= fish).fit()

# Açık interaction : (Explicit)

mdl_mass_vs_both_explicit = ols("mass_g ~ length_cm + species + length_cm : species",
                                data = fish).fit()

# Bütün explanaotry variableları + ile ayırıp sonra üçüncü terim olan iki explanatory variableları colummn ile ayırmaktır 


# Easier to understand coefficients : 

mdl_mass_vs_both = ols("mass_g ~ length_cm * species",
                       data =fish).fit()

print(mdl_mass_vs_both.params) # Burada parametreleri anlamak zor çünkü implicit bu aslında

mdl_mass_vs_both_inter = ols("mass_g ~ speciess + species:length_cm + 0",
                             data = fish).fit() # Burda ise ufak bir data manipulation yaparak daha okuanabilir hale getirdik biz columnlarımızı

print(mdl_mass_vs_both_inter.params)

# Eğer ki biz bunları yine ayrı datasetlere ayırıp yine parametrelerine baksaydık yine aynı değerleri görecektik
