import pandas as pd
import numpy as np
import statistics
from scipy.stats import uniform
import matplotlib.pyplot as plt
from scipy.stats import binom

# Not : Girilen parameterler olasılıkta her zaman daha azı kale alınarak yapılır yani eğerki biz .cdf() içine bir değer yazarsak bu her zaman azı olarak görülür büyüğü için birden çıkartılır

# Binomial Distribution aslında iki tane outcome demektir yani yaşadığımız olayda outcome 2 olmak zorunda daha fazla olamaz

# Outcome değerlerini 1 ve 0 olarak seçebiliriz

one_coin = binom.rvs(1, 0.5, size = 3) # Burda bir tane coini 3 kere atmamız isteniyor bizden (Atılcak coin sayısı, ihtimali, kaç kere atacağımız)

print(one_coin)

three_coin = binom.rvs(3, 0.5, size = 10) # Burda göreceğimiz 2,3 değerleri 3 tane paradan kaçında tura olduğudur yani başarı sayısıdır

print(three_coin)

# Bunun yanında hileli para da kullanılabilir 

unfair_coin = binom.rvs(3, 0.25, size = 5) # burda yazı tarafı daha ağır yani tura gelme olasılığı 0.25 sadece olasılığını değiştirdik 

# İngilizce tanımı ise : 

""" 
Probability distribution of the number of successes in a sequence of independent trials

We uses 2 parameters : 

n : total number of trials

p : probability of success

binom.rvs(n = x, p = y, size = z)

"""

# Finding probability of a sequence 

# Burada binomial olan bir olayın olma olasılığını bulacağız : 

binom.pmf(7, 10 ,0.5) # PMF : probability mass function

prob_of_heads = binom.pmf(7,10,0.5) # Burdaki sıralama şudur kaç tane tura istiyoruz, kaç kere para atıyoruz, tura gelme olasılığı

print(prob_of_heads)

# Peki ya 7 ya da daha az tura gelme olasılığı nedir ? : 

binom.cdf(7,10,0.5) # CDF : cumulative distribution function

prob_of_fewer_heads = binom.cdf(7,10,0.5)

print(prob_of_fewer_heads)

# 7 den daha fazla gelme olasılıgı ise : 

1 - prob_of_fewer_heads 

print(1 - prob_of_fewer_heads)

# Expected Value of Binomial Distribution : 

""" 
Expected value : n x p yani -> trials x probability of heads or tail
"""

# Binomial Distribution olması için sonuçlar birbirlerini etkilememeli independent olmalı


