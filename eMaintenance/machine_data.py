import numpy as np
import pandas as pd

# Lets-plot can be made to work in notebook, bug reported
# https://youtrack.jetbrains.com/issue/PY-70557/Lets-plot-does-not-work-from-WSL2
import lets_plot as lplt
import matplotlib.pyplot as plt

from scipy.stats import weibull_min, norm

#%%
# Read the data file into a dataframe

df = pd.read_csv('machine_data.csv')

#%%
# Data re-conditioning

df.drop(columns='Unnamed: 0', inplace=True)
df.loc[df.manufacturef == 'c', 'manufacturef'] = 'C'
df.loc[:, 'manufacturer'] = df.manufacturef
df.drop(columns='manufacturef', inplace=True)

#%%
# Print basic statistics

bounds_df = df.groupby('manufacturer').agg(  # describe()
    {
        "time": ["min", "max", "median", "mean"],
        "load": ["min", "max", "median", "mean", "var"],
    }
)
print(bounds_df.T)
print("     mode     ", end="")
for _name, dfa in df.groupby('manufacturer'):
    load = dfa[['load']]
    print(f"{' & '.join(map(lambda v: str(int(v[0])), load.round().mode().values)):11s}", end='')

#%%
# Prepare for plotting

lplt.LetsPlot.setup_html()

#%%

load_box = (lplt.ggplot(df)
           + lplt.geom_boxplot(lplt.aes(x='manufacturer', y='load',
                                        color='manufacturer', fill='manufacturer'),
                               outlier_shape=21, outlier_size=1.5, size=2,
                               alpha=.5, width=.5, show_legend=False)
           )
lplt.ggsave(load_box, "load_manufacturer_boxplot.svg")
load_box.show()

time_box = (lplt.ggplot(df)
           + lplt.geom_boxplot(lplt.aes(x='manufacturer', y='time',
                                        color='manufacturer', fill='manufacturer'),
                               outlier_shape=21, outlier_size=1.5, size=2,
                               alpha=.5, width=.5, show_legend=False)
           )
lplt.ggsave(time_box, "time_manufacturer_boxplot.svg")
time_box.show()

#%%
from math import floor, ceil
plot_minimum = df.min(numeric_only=True).apply(lambda x: floor(x - 0.5))
plot_maximum = df.max(numeric_only=True).apply(lambda x: ceil(x + 0.5))

grpByManu = df.groupby(['manufacturer'], sort=True)

relation = (lplt.ggplot(df)
 + lplt.geom_point(lplt.aes(x='load', y='time', color='manufacturer'))
 + lplt.ggtitle("Relation between load and time")
 )

relation.show()
lplt.ggsave(relation, "manufacturer_relation.svg")

#%%

from scipy.optimize import curve_fit


def generic_fn(x, a, b, c, d):
  return (a * x + b) / (c * x + d)


# fit_relation = lplt.ggplot()
fit_relation = relation
plot_range = np.linspace(plot_minimum.load, plot_maximum.load, 20)
for name, dfa in grpByManu:
  popt, pcov = curve_fit(generic_fn, dfa.load, dfa.time)
  fit_relation = fit_relation + lplt.geom_line(lplt.aes(x=plot_range, y=generic_fn(plot_range, *popt)))
  print(f"{name[0]}: time = ({popt[0]:.3f} * load + {popt[1]:.3f}) / ({popt[2]:.3f} * load + {popt[3]:.3f})")

lplt.ggsave(fit_relation, "manufacturer_fit_relation.svg")
fit_relation.show()

#%%

fig, axs = plt.subplots(3, 2, sharex=False, sharey=False)
for ax in axs[:, 0]:
  ax.set(xlim=(plot_minimum['load'], plot_maximum['load']))
for ax in axs[:, 1]:
  ax.set(xlim=(plot_minimum['time'], plot_maximum['time']))

for index, (name, dfa) in enumerate(grpByManu):
  name = "Manufacturer " + name[0]

  load = dfa['load']
  time = dfa['time']

# %%
bins = 20
n, bins, patches = axs[index, 0].hist(load, bins=bins, label=name)
axs[index, 0].title.set_text(f"Histogram of load distribution")
axs[index, 0].legend()
load_mean, load_std = norm.fit(load)


def load_cdf(range):
  return load.size * norm.cdf(range, load_mean, load_std)


def plot_cdf(ax, range, cdf):
  left = range[:-1]
  right = range[1:]
  ax.plot((left + right) / 2, (cdf(right) - cdf(left)))


plot_cdf(axs[index, 0], bins, load_cdf)

# %%
bins = 20
n, bins, patches = axs[index, 1].hist(time, bins=bins, label=name)
axs[index, 1].title.set_text(f"Histogram of time distribution")
axs[index, 1].legend()

c, loc, scale = weibull_min.fit(time)


def time_cdf(range):
  return time.size * weibull_min.cdf(range, c=c, loc=loc, scale=scale)


plot_cdf(axs[index, 1], bins, time_cdf)

plt.show()

plt.savefig("manufacturers_hist_fit.svg", dpi=150)

#%%
# An in report unused plot

man_hist = lplt.ggplot(df, lplt.aes(x='load', fill='manufacturer'))
man_hist = (man_hist
 + lplt.ggsize(700, 300)
 + lplt.scale_fill_brewer(type='seq')
 + lplt.geom_histogram(position='dodge', alpha=0.7)
 + lplt.theme(panel_grid_major_x='blank')
 + lplt.geom_vline(lplt.aes(xintercept=bounds_df.load['mean'],
                            color=bounds_df.index.values),
                   linetype="dashed")
)
lplt.ggsave(man_hist, "load_manufacturer_histogram.svg")
man_hist.show()
