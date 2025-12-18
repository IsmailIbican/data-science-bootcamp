import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


# Normal distribution 'bell curve' yani 'çan eğrisi' tarzda bir grafiğe sahiptir diğerlerine nazaran

# Simetriktir grafik

# Eğri hiç bir zaman 0'a dokunmaz

# Mean ve Standart Deviation kullanılarak oluşturulur

# Mean peak noktasıdır grafiğin

# Mean = 0 ve Std = 1 ise bu standart normal distribution olur

# Bunu kullanarak diğer grafiklerdeki dataların ortak değerlerini bulup bir grafik haline getirebiliriz 'bell curve' şeklinde

# Percentage bulabiliriz bununla birlikte


# Elimizde kadınların boy grafikleri olsun ve bu grafiğe göre bir işlem yapalım (assume et)

# Sorumuz ise şu boyu 154 cm'den kısa olan kadınların yüzdesi

norm.cdf(154,161, 7) # Buradaki değerler şöyle sıralanıyor (üst sınır, mean yani ortalaması, std yani standart sapması)

# Eğer ki uzun olanları bulmak istiyorsak

1 - norm.cdf(154,161,7) # Bu arada std'yi eğer ki data setimiz olsaydı np.std(NHANES['womans_heights_from_survey']) yaparak bulabilirdik ve biz de 7 bulurduk

# Bir aralıkta bulmak istersek eğer : 

# 154-157 arasındaki kadınların yüzdeliğini hesaplamak için : 

norm.cdf(157,161,7) - norm.cdf(154,161,7)

# Ve biz buradan uzunluk da bulabiliriz percentage ile 

# What height are 90% of women shorter than  (Hangi boydan kısa olan kadınlar tablonun 90%'sini oluşturur)

norm.ppf(0.9, 161, 7)

# Eğer ki hangi boydan uzun olanlar tablonun 90% sini oluşturur bunu bulmak istersek : 

norm.ppf((1-0.9), 161, 7) # Çıkarma işlemi zorunlu 

# Bunda da random veriler oluşturabiliriz bizi mean ve std'ye uyan 

random_variable_sampling = norm.rvs(161,7, size = 10) # mean ve std'si 161 e 7 olan 10 tane random değer çağır demek bu 

print(random_variable_sampling)
