import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt

# Correlation coefficent :

# Quantifies the linear relationship between two variables

# Number between -1 and 1

# Magnitude corresponds to strength of relationship

# Sign corresponds to direction of relationship

# Eğer ki magnitude 1 e yakın ise bu güçlü bir relation ship demektir ya da -1 

# Bu değer 0' a yaklaştıkça relation zayıflar ve no relatinoship anlamına gelir

# No relationship demek aslında x değerlerimizin y hakkında bize herhangi bir bilgi vermediğidir

# Sign + -> x artarken y de artar 

# Sign - -> x artarken y azalır

# Ve biz genelde correlationları scatterplot üstünde gösteririz bunun için de seaborn bizim için biçilmiş bir kaftandır

msleep = pd.read_csv('mammal_sleep_data.csv')

sns.scatterplot(x = 'awake', y= 'bodywt', data = msleep)

plt.show()

# Lineer bir doğru geçirmek istersek eğer ki scatterplottan bu fonksiyonu kullanacağız : 

sns.lmplot(x = "awake", y="bodywt", data = msleep, ci = None)

sns.lmplot(x = "awake", y="bodywt", data = msleep, ci = 95)

plt.show()

# ci = bizim buradaki güven aralığımmızdır aslında bunu çizerek bir güven aralığı çizilir regresyon çizgisi etrafına

# Eğer ki biz iki sütun arasındaki correlation coefficient değerini bulmak istersek böyle yapacağız : 

msleep['awake'].corr(msleep['bodywt'])

print(msleep['awake'].corr(msleep['bodywt']))

# Aralarındaki correlation bulurken yazdığımız yerler önemli değildir x ile y nin y ile x in correlation değerleri her zaman aynıdır

""" 
Ways to calculate correlation : 

Pearson product-moment correlation : (We used this calculation to calculate correlation)

Kendall's tau
Speaman's rho
"""

