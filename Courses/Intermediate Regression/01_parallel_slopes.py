import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from statsmodels.formula.api import ols
from statsmodels.formula.api import logit
from statsmodels.graphics.mosaicplot import mosaic


# Bu konuda biz birden fazla explanatory variable ile birlikte çalışmayı öğreneceğiz

# Birden fazla explanatory variable bize daha iyi bir bakış açısı ve prediction şansı sunar

# 4 chapterdan oluşacak bu konu : 

""" 
Chapter 1 : 

Parallel slopes regression : 

Bu linear regression için özel bir durumdur bir numerical explanatory variable ve bir categorical explanatory variable vardır burada

Chapter 2 :

Interactions 
Simpon's Paradox

Chapter 3 :

More explanatory variables

How linear regression works

Chapter 4 : 

Multiple logistic regression 

The logistic distribution

How logistic regression works
"""

fish = pd.read_csv("fish_data.csv")

# Biz bir tane explanatory variable için ols fonksiyonunu kullanıyorduk 

mdl_mass_vs_length = ols("mass_g ~ length_cm",
                         data = fish).fit()

print(mdl_mass_vs_length.params) # Shows intercept(coefficient) and slope

# Bir tane categorical variable ile çalışıyorsak eğer ki : 

mdl_mass_vs_species = ols("mass_g ~ species + 0",
                          data = fish).fit()

print(mdl_mass_vs_species.params) # Her bir species değeri için biz bir tane intercept değeri aldık burada 


# Peki ikisini aynı anda yapmak istersek ne yapacağız ? : 

mdl_mass_vs_both = ols("mass_g ~ length_cm + species + 0",
                       data = fish).fit()

print(mdl_mass_vs_both.params) # Her bir species değerinin interceptleri ve total slope değerini alıcaz biz burada

# Explanatory variable için Visualization (Numerical) : 

sns.regplot(x = "length_cm",
            y = "mass_g",
            data = fish,
            ci = None)

plt.show()

# Categorical Explanatory Variable Visualization 1 : 

sns.boxplot(x = "species",
            y = "mass_g",
            data = fish,
            showmeans = True) # Burada mean değerlerini de gösteren bir boxplot oluşturduk

plt.show()

# Seaborn ikisini birlikte visualize etmekte iyi bir kütüohane değil o yüzden kendimiz yapıcaz 

# Visualization : both explanatory variables 

coeffs = mdl_mass_vs_both.params

ic_bream, ic_hamsi, ic_palamut, ic_perch, ic_pike, ic_roach, sl = coeffs

sns.scatterplot(x = "length_cm",
                y = "mass_g",
                hue = "species",
                data = fish) # Burada normal anam babam length ile mass arasındaki ilişkiyi gösterdik

# Burada ise her bir intercept değeri farklı olan species değerleri için aralarındaki ilişkileri gösteren doğrular çeker yaptıkları davranışlara bakıcaz

plt.axline(xy1=(0,ic_bream), slope = sl, color = "blue")
plt.axline(xy1=(0,ic_hamsi), slope = sl, color = "purple")
plt.axline(xy1=(0,ic_palamut), slope = sl, color = "brown")
plt.axline(xy1=(0,ic_perch), slope = sl, color = "orange")
plt.axline(xy1=(0,ic_pike), slope = sl, color = "green")
plt.axline(xy1=(0,ic_roach), slope = sl, color = "red")

# Her bir eğri birbirine paraleldir

plt.show()