import pandas as pd 
import numpy as np
from scipy.stats import iqr
import statistics
import seaborn as sns
import matplotlib.pyplot as plt 

# Probability distribution

# Her bi olası sonucun ihtimalini tanımlar

# Expected value : mean of a probability distribution(Ortalama değer)

# Probability'i biz barplot'ta alan hesaplayarak bulabiliriz

# Eğer ki bizim elimiz de bulunan her bir probability değeri aynı ise bu Discrete uniform distribution olur

die = pd.read_csv('dice_data.csv')
print(die)

np.mean(die['number']) # number sütunun ortalama değeri

print(np.mean(die['number'])) # Bu expected Value

rolls_10 = die.sample(10, replace= True) # replacement yaparak 10 kere random zar attık burada

print()
print(rolls_10)

# Visualizing a sample

# bu matplotlib ile çizildi
rolls_10['number'].hist(bins = np.linspace(1,7,7)) # linspace zaten biliyoruz 1 den 7  ye kadar 7 dahil 6 aralığa bööl demek bu sadece bir array oluşturur bins ises bunu histgorama dokecek bir parametre
plt.show()

# bu da seaborn ile çizildi

bins = np.linspace(1,7,7)

sns.histplot(data= rolls_10, x = 'number', bins = bins, kde= False, edgecolor = 'black') # kde yogunluk çizgisi çeker 
plt.show()

# Sample distribution -> Random Theoretical probability distribution -> Flat 

# Law of large numbers : Sample size büyüdükçe expected value değerine yaklaşılır