import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Set seed to 321

np.random.seed(321)

sample_means = []

# Loop 30 times to take 30 means

for i in range(30) : 
    
    # Take sample of size 20 from num_users col of all_deals with replacement
    
    cur_sample = all_deals['num_users'].sample(20, replace = True)
    
    # Take mean of cur_sample
     
    cur_mean = np.mean(cur_sample)
     
    # Append cur_mean to sample_means
    sample_means.append(cur_mean)  
      
      
# Print mean of sample_means
print(np.mean(sample_means))
    
    
    
# Print mean of num_users in amir_deals
print(np.mean(amir_deals['num_users']))

# Create a histogram num users of amir deals and show it
amir_deals['num_users'].hist()
plt.show()

# Set seed to 104
np.random.seed(104)

# Sample 20 num_users with replacement from amir_deals

samp_20 = amir_deals['num_users'].sample(20, replace= True)

# Take mean of samp_20

np.mean(samp_20)


sample_means_of = []

# Loop 100 times

for i in range(100) : 
    
    # Take sample of 20 num_users
    
    samp_20 = amir_deals['num_users'].sample(20,replace=True)
    
    # Calculate mean of samp_20
    
    samp_20_mean = np.mean(samp_20)
    
    # Append it 
    
    sample_means_of.append(samp_20_mean)
    
print(sample_means_of)

# Convert to series and show histogram

sample_means_series = pd.Series(sample_means_of)
plt.hist(sample_means_series)
plt.show()

    