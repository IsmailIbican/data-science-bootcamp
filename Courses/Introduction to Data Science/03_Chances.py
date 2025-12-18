import pandas as pd
import numpy as np
from scipy.stats import iqr
import statistics

# Measuring chance : 

sales_counts = pd.read_csv('sales_count_data.csv')

print(sales_counts)

sales_counts.sample() # Random bir tane eleman seçer tablomuzdan

print()
print(sales_counts.sample()) # Her seferinden farklı bir değer alırız

# Setting a random seed

np.random.seed(10) #Eğer biz bu parantezin içine büyüklüğü fark etmeksizin bir sayı yazarsak en son elde edittiğimiz sample verisi her zaman bize gelmeye devam edicek random şekilde çağırsak bile

print() 
print(sales_counts.sample()) # Yukarıdaki sample dan claire geldi ve her çalıştığında claire gelmeye devam edecek

# Second choose (Sampling without replacement) 

# Şimdi claire bizim o 4 kişi arasından seçildi ve geriye 3 kişi kaldı claire değerini almadan nasıl 3 kişi arasında seçim yapıcaz onu bulacağız

sales_counts.sample(2) # 4'lü arasından 2 sini seç demek yani

print()
print(sales_counts.sample(2))

# Burada aslında böyle bir seçim yolu da izleyebiliriz : 

sample_without_claire = sales_counts[sales_counts['name'] != 'Claire'].sample(n = 2, random_state=42) # Burada filtreleme yaptık tablomuzda ve sample methodunun içini doldurduk random_state np.random.seed ile aynı işlevi görmekte

# Sampling with replacement 

# Burda bir kişiyi seçtikten sonra geri koymamız lazım aldığımız yere yani birisini seçtiğimiz de ki olasılık hiç bir zaman değişmemeli

sales_counts.sample(5, replace= True) # 5 kişiyi yerlerine geri koyarak rastgele seç demek

# Independent events

# Sampling with replacement yaparken genelde aslında her bir olay independent olur mesela ilk başta brian seçtik %25 ihtimalle 2.de de claire seçtik o da %25 ihtimalle ilk olay 2. olayı etkilemedi
    


