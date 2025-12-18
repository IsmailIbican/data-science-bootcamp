import pandas as pd
import numpy as np
from scipy.stats import binom
from scipy.stats import norm


# Simulate a single deal (Burada bizden bir tane dealın gerçekleşip gerçekleşmeyeceğini soruyor prob = %30)

print(binom.rvs(1,0.3, size = 1))

# Simulate 1 week of 3 deals (Bir haftada toplam 3 deal yapıyor)

print(binom.rvs(3,0.3, size = 1))

# Simulate 52 weeks of 3 deals (1 yılda her hafta 3 deal yapıyor)

deals = binom.rvs(3, 0.3, size = 52)

# Print mean deals won per week

print(np.mean(deals))


# Probability of closing 3 out of 3 deals (Her bir dealı başarıyla tamamlama olasılığı soruluyor)

prob_3 = binom.pmf(3,3,0.3)

print(prob_3)

# Probability of closing <= 1 deal out of 3 deals

prob_less_than_or_1 = binom.cdf(1,3,0.3)

print(prob_less_than_or_1)

# Probability of closing > 1 deal out of 3 deals

prob_greater_than_1 = 1 - binom.cdf(1,3,0.3)

print(prob_greater_than_1)


# Find the expected value of 0.25 win rate 0.30 win rate and 0.35 win rate of amir

won_30pct = 3 * 0.3
won_25pct = 3 * 0.25
won_35pct = 3 * 0.35