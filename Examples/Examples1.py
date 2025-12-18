import pandas as pd 
import numpy as np 
from scipy.stats import iqr
import statistics


# Count the deals for each product 

counts = amir_deals['product'].value_counts() # value_counts() methodu x sütunundaki her bir değerin kaç kere geçtiğini gösterir(benzersiz her bir değerin)
print(counts)

# Calculate probability of picking a deal with each product

probs = counts / amir_deals.shape[0] # Burda amirin kaç kere deal'a gittigin gösteriyor .shape[0] (Toplam kaç tane satır oldugunu gösterir bize .shape[0])

# Set random seed to 24

np.random.seed(24)

# Sample 5 deals without replacement 

sample_without_replacement = amir_deals.sample(5)
print(sample_without_replacement)

# Sample 5 deals with replacement 

sample_with_replacement = amir_deals.sample(5, replace = True)
print(sample_with_replacement)
