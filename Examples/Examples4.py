import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Histogram of amount with 10 bins and show plot(amir_deals var bins = 10)

amir_deals['amount'].hist(bins = 10)
plt.show()

# Probability of deal < 7500

prob_less_7500 = norm.cdf(7500,5000,2000)

# Probability of deal > 1000

prob_over_1000 = 1 - norm.cdf(1000,5000,2000)

# Probability of deal between 3000 and 7000

prob_3000_to_7000 = norm.cdf(7000,5000,2000) - norm.cdf(3000,5000,2000)

# Calculate amount that 25% of deals will be less than

pct_25 = norm.ppf(0.25, 5000, 2000)

# Calculate new average amount(mean = 5000, std = 2000 , mean increased %20)

new_mean = 5000 + (5000 * 0.2)

# Calculate new std (std increased 30%)

new_std = 2000 + (2000 * 0.3)

# Simulate 36 new sales

new_sales = norm.rvs(new_mean, new_std, size = 36)

# Create histogram than show it

plt.hist(new_sales)
plt.show()


