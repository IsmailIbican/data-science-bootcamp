import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a histogram of group_size of restaurant_groups and show plot (bins = [2,3,4,5,6])

restaurant_groups['group_size'].hist(bins = [2,3,4,5,6]) 
plt.show()

# Create probability distribution

size_dist = restaurant_groups['group_size'].value_counts() / restaurant_groups.shape[0] 

# Reset index and rename columns

size_dist = size_dist.reset_index()
size_dist.columns = ['group_size', 'prob']

# Biz bu işlemi böyle de yapabiliriz : 

size_dist = size_dist.to_frame(name = 'probability') # bu da aslında indexi sıfırlayıp yeni column degerleri oluşturmak gibidir.

# Ya da böyle de yapabiliriz. 

size_dist = restaurant_groups['group_size'].value_count(normalize = True).reset_index() # Bu da ilk yaptıgımızın bir başka alternatifi


print(size_dist)

# Calculate expected value 

expected_value = np.sum(size_dist['group_size'] * size_dist['prob']) # Burada yapmam gereken soruda verilmişti zaten expected value için 

print(expected_value)

# Subset groups of size 4 or more

groups_4_or_more = size_dist[size_dist['group_size'] >= 4] # Burada filtreleme işlemi var 4 ya da daha fazla demiş tabloya gidicez tablodaki spesifik sütunu seçip şarta göre karşılaştıracağız

# Sum the probabilities of groups_4_or_more

prob_4_or_more = np.sum(groups_4_or_more['prob'])

print(prob_4_or_more)


