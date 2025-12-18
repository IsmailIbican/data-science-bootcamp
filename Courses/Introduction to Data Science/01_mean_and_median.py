import numpy as np 
import pandas as pd
import statistics
import matplotlib.pyplot as plt

# Finding mean : 
msleep = pd.read_csv("mammal_sleep_data.csv")
print(np.mean(msleep['awake']))

# Finding median : 

msleep['awake'].sort_values() # We have to sort all values to find median 

# Imagine that we have 41 indexed datas ;  

msleep['awake'].sort_values().iloc[41] # Iloc function manipulate integer datas to find specific rows or columns etc

print(np.median(msleep['awake']))

# Finding mode : 

# And also while finding mode we don't need only numerical data. 

msleep["awake"].value_counts() # Count the values and find the which has mostly repeated

print(statistics.mode(msleep["awake"])) # Finding awake column's mode 

print(statistics.mode(msleep["vore"])) # Finding 'vore' column's mode

# Adding an outliner : 

msleep[msleep['vore'] == 'insecti'] # subset msleep to select rows where 'vore' equals to 'insecti'

msleep[msleep['vore'] == 'insecti']['awake'].agg([np.mean, np.median]) # find the mean and median of where 'vore' equals to 'insecti' of awake


# Histogram of values : 

msleep['values'].hist() # it gives us a histogram 

# Showing plot : 

plt.show()




