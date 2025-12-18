import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# 5 kere zar atalım öncelikle ve mean yani ortalama değerlerine bakalım

die = pd.Series([1,2,3,4,5,6])

die_5 = die.sample(5, replace = True)

print(np.mean(die_5))

# şimdi 10 kere 5 kere zar atıp mean alma işlemini yapalım : 

sample_means = []

for i in range(10) : # Burda demek istiyor ki 10 kere tekrar et bu işlemi
    samp_5 = die.sample(5, replace= True)
    sample_means.append(np.mean(samp_5))

print(sample_means)

plt.hist(sample_means)
plt.show()

# 100 sample means yapalım şimdi de : 

sample_means_100 = []

for i in range(100) : 
    sample_means_100.append(np.mean(die.sample(5, replace=True)))

plt.hist(sample_means_100)
plt.show()

# Şimdi de 1000 tanesinin means'ini bulalım : 

sample_means_1000 = []

for i in range(1000) : 
    sample_means_1000.append(np.mean(die.sample(5, replace=True)))
    
plt.hist(sample_means_1000)
plt.show()

# Sample sayısı büyüdükçe bizim grafiğimiz normal distribution'a benzemeye başladı.

# Biz buna central limit theorem deriz yani yaklaşma teoremi yani bi nevi limit

# Mesela 10k sample örneği verelim birlikte 

sample_means_10k = []

for i in range(10000) : 
    sample_means_10k.append(np.mean(die.sample(5, replace=True)))
    
plt.hist(sample_means_10k)
plt.show()

# Samplelar random ve independent olmalı central limit theorem de 


# Standart deviation and the CLT

sample_sds = []

for i in range(1000) : 
    sample_sds.append(np.std(die.sample(5, replace=True)))

plt.hist(sample_sds)
plt.show()

# Proportion and the CLT

sales_team = pd.Series(['Amir', 'Brain', 'Claire', 'Damian'])

print(sales_team.sample(10,replace=True))

# Bu methodlar eğer ki elimizde çok fazla veri varsa ve bu verilerdeki dataları elde edecek yeteri kadar zamanımız yoksa bir kaç tane sample alıp yakın değerleri bulmada yardımcı olur





    
    
    