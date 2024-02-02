#%%

import numpy as np
import pandas as pd

import matplotlib
import lets_plot as lplt
import matplotlib.pyplot as plt
#matplotlib.use("TkAgg")

#%% mathplotlib notebook

'''
Submit your solutions in pdf format, with code and plots supporting your answers.
machine_data contains raw data of a part from 3 manufactures A, B, C
The system is run to failure under load
The load and the operation time is provided in each row

What is the range of load and time during operation for each manufacturer?
What is the most expected load value?
How are the load and time related?
Which distribution best describes the load?
Which distribution best describes the time?

Which manufacturer has the best performance and why?

'''
#%%
# read the data file into a dataframe
df = pd.read_csv('machine_data.csv')
print(df)

print(df.shape)

#%% 
"""
Drop the index
"""
df.drop(columns='Unnamed: 0', inplace=True)
minimum = df.min()
maximum = df.max()
#%%
"""
Extract data for a given manufacturer
"""
lplt.ggplot(df)
lplt.show()

grpByManu = df.groupby(['manufacturef'])

fig, axs = plt.subplots(3+1, 1)
ax, axs = axs[0], axs[1:]
ax.axis([minimum['load'], maximum['load'], minimum['time'], maximum['time']])
ax.set_title("Relation between load and time")
ax.set_xlabel("Load")
ax.set_ylabel("Time")

for index, (name, dfa) in enumerate(grpByManu):
    name = "Manufacturer " + name[0]

#%%

    load = dfa['load']
    time = dfa['time']

    #%%
    '''
    Is there a relationship between load and time
    '''
    path_collection = ax.scatter(load, time)
    path_collection.set_label(name)

    #%%
    '''
    Characteristics of data
    mean, median, mode
    '''
    print(f"{name}, load mean={dfa['load'].mean()}, median={dfa['load'].median()}, mode={dfa['load'].mode()}")
    #%%
    '''
    How is load distributed
    Why does it matter
    uniform, normal, exponential, weibull
    '''
    x = dfa[['load']].plot(ax=axs[index], kind='hist', bins=10)
    x.set_label(name)
    x.set_title(f"Histogram of load distribution")

    #%%
    '''
    variance, standard deviation
    What is the meaning of 6sigma
    '''
    print(f"{name}, load var={dfa['load'].var()}, stddev={dfa['load'].std()}")
    #%%
    '''
    Other plots that can be useful 
    boxplot
    '''
#%%
ax.legend()
plt.show()
