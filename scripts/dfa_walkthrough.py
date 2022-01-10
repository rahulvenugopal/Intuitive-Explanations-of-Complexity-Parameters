#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 11:07:55 2021
- DFA tutorial based on nolds implementation
@author: Rahul Venugopal

References
- https://nolds.readthedocs.io/en/latest/nolds.html#detrended-fluctuation-analysis
- https://nolds.readthedocs.io/en/latest/_modules/nolds/measures.html#dfa
- Detrended fluctuation analysis: a scale-free view on neuronal oscillations
https://www.frontiersin.org/articles/10.3389/fphys.2012.00450/full

"""

# Run the custom fucntions in the following cell block
#%% Helper functions
import warnings

def poly_fit(x, y, degree, fit="RANSAC"):
  # check if we can use RANSAC
  if fit == "RANSAC":
    try:
      # ignore ImportWarnings in sklearn
      with warnings.catch_warnings():
        warnings.simplefilter("ignore", ImportWarning)
        import sklearn.linear_model as sklin
        import sklearn.preprocessing as skpre
    except ImportError:
      warnings.warn(
        "fitting mode 'RANSAC' requires the package sklearn, using"
        + " 'poly' instead",
        RuntimeWarning)
      fit = "poly"

  if fit == "poly":
    return np.polyfit(x, y, degree)
  elif fit == "RANSAC":
    model = sklin.RANSACRegressor(sklin.LinearRegression(fit_intercept=False))
    xdat = np.asarray(x)
    if len(xdat.shape) == 1:
      # interpret 1d-array as list of len(x) samples instead of
      # one sample of length len(x)
      xdat = xdat.reshape(-1, 1)
    polydat = skpre.PolynomialFeatures(degree).fit_transform(xdat)
    try:
      model.fit(polydat, y)
      coef = model.estimator_.coef_[::-1]
    except ValueError:
      warnings.warn(
        "RANSAC did not reach consensus, "
        + "using numpy's polyfit",
        RuntimeWarning)
      coef = np.polyfit(x, y, degree)
    return coef
  else:
    raise ValueError("invalid fitting mode ({})".format(fit))

# Logarithmic
def logarithmic_n(min_n, max_n, factor):
  """
  Creates a list of values by successively multiplying a minimum value min_n by
  a factor > 1 until a maximum value max_n is reached.

  Non-integer results are rounded down.

  Args:
    min_n (float):
      minimum value (must be < max_n)
    max_n (float):
      maximum value (must be > min_n)
    factor (float):
      factor used to increase min_n (must be > 1)

  Returns:
    list of integers:
      min_n, min_n * factor, min_n * factor^2, ... min_n * factor^i < max_n
      without duplicates
  """
  assert max_n > min_n
  assert factor > 1
  # stop condition: min * f^x = max
  # => f^x = max/min
  # => x = log(max/min) / log(f)
  max_i = int(np.floor(np.log(1.0 * max_n / min_n) / np.log(factor)))
  ns = [min_n]
  for i in range(max_i + 1):
    n = int(np.floor(min_n * (factor ** i)))
    if n > ns[-1]:
      ns.append(n)
  return ns

# plot_reg

def plot_reg(xvals, yvals, poly, x_label="x", y_label="y", data_label="data",
             reg_label="regression line", fname=None):
  """
  Helper function to plot trend lines for line-fitting approaches. This
  function will show a plot through ``plt.show()`` and close it after the window
  has been closed by the user.

  Args:
    xvals (list/array of float):
      list of x-values
    yvals (list/array of float):
      list of y-values
    poly (list/array of float):
      polynomial parameters as accepted by ``np.polyval``
  Kwargs:
    x_label (str):
      label of the x-axis
    y_label (str):
      label of the y-axis
    data_label (str):
      label of the data
    reg_label(str):
      label of the regression line
    fname (str):
      file name (if not None, the plot will be saved to disc instead of
      showing it though ``plt.show()``)
  """
  # local import to avoid dependency for non-debug use
  import matplotlib.pyplot as plt
  plt.plot(xvals, yvals, "bo", label=data_label)
  if not (poly is None):
    plt.plot(xvals, np.polyval(poly, xvals), "r-", label=reg_label)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.legend(loc="best")
  if fname is None:
    plt.show()
  else:
    plt.savefig(fname)
  plt.close()

#%% DFA calculation

# Create a sample data
import numpy as np
data = np.random.random(1000)

# Initial parameters
nvals = None
overlap = True
order = 1
fit_trend = "poly"
fit_exp = "RANSAC" # robust to outliers
debug_plot = False
debug_data = False
plot_file = None

# Converting data to an array
data = np.asarray(data)
total_N = len(data)

# Fixing the window lengths for data having more than 100 data points
nvals = logarithmic_n(4, 0.1 * total_N, 1.2)

# Cumulative sum of deviations from mean
walk = np.cumsum(data - np.mean(data))

# initialise
fluctuations = []

# Looping through different window sizes to capture standard deviations
for n in nvals:
  # subdivide data into chunks of size n
  # step size n/2 instead of n for overlap
  d = np.array([walk[i:i + n] for i in range(0, len(walk) - n, n // 2)])
  # each row of d is data slice shifted with 50% overlap

  # calculate local trends as polynomes
  x = np.arange(n)

  # fitting a regression line of order 1
  tpoly = [poly_fit(x, d[i], order, fit=fit_trend)
           for i in range(len(d))]
  tpoly = np.array(tpoly)
  # tpoly has intercept and slope

  # find the trend line
  trend = np.array([np.polyval(tpoly[i], x) for i in range(len(d))])
  # calculate standard deviation ("fluctuation") of walks in d around trend
  flucs = np.sqrt(np.sum((d - trend) ** 2, axis=1) / n)
  # calculate mean fluctuation over all subsequences
  f_n = np.sum(flucs) / len(flucs)
  fluctuations.append(f_n)

fluctuations = np.array(fluctuations)


# filter zeros from fluctuations
# I think this is to avoid the logarithm issues with zero
nonzero = np.where(fluctuations != 0)
nvals = np.array(nvals)[nonzero]
fluctuations = fluctuations[nonzero]

if len(fluctuations) == 0:
  # all fluctuations are zero => we cannot fit a line
  poly = [np.nan, np.nan]
else:
  poly = poly_fit(np.log(nvals), np.log(fluctuations), 1,
                  fit=fit_exp)
print(poly[0])

#%% Visualising polynomial fit demo
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12345)
y = np.random.random(500)
x= range(len(y))

trend = np.polyfit(x,y,1)
plt.plot(x,y,'o')
trendpoly = np.poly1d(trend)
plt.plot(x,trendpoly(x))

plt.plot(x,y,color = 'steelblue')
plt.title('Random signal of length 1000 data points')
ax = plt.gca()
# ax.set_xlim([xmin, xmax])
ax.set_ylim([-0.5, 2])

#%% Plotting segments
plt.plot(np.transpose(d))

plt.plot(np.transpose(trend))

plt.plot(np.transpose(d-trend))

ax = plt.gca()
# ax.set_xlim([xmin, xmax])
ax.set_ylim([-4, 4])
