import pandas as pd
import numpy as np
from scipy.stats import poisson


# Probability of 5 responses (amir responds 4 deals average in a day)

prob_5 = poisson.pmf(5,4)

# Probability of 5 responses (5.5 average on a day)

coworker_prob_5 = poisson.pmf(5,5.5)

# Probability of 2 or fewer responses

prob_less_than_2 = poisson.cdf(2, 4)

# Probability of > 10 responses

prob_greater_than_10 = 1 - poisson.cdf(10, 4)