# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 18:16:43 2022
- Generate 10 waveform of different nature
- Run complexity analysis
- Compare, contrast

To Do
- Compare with more waveforms and parameter space for each complexity variable
@author: Rahul Venugopal
"""
#%% Loading libraries
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import random
from random import gauss
import antropy as ant
import seaborn as sns
import pandas as pd

#%% Signal generator
def generate_signal(length_seconds, sampling_rate, frequencies_list, func="sin", add_noise=0, plot=True):
    r"""
    Generate a `length_seconds` seconds signal at `sampling_rate` sampling rate. See torchsignal (https://github.com/jinglescode/torchsignal) for more info.

    Args:
        length_seconds : int
            Duration of signal in seconds (i.e. `10` for a 10-seconds signal)
        sampling_rate : int
            The sampling rate of the signal.
        frequencies_list : 1 or 2 dimension python list a floats
            An array of floats, where each float is the desired frequencies to generate (i.e. [5, 12, 15] to generate a signal containing a 5-Hz, 12-Hz and 15-Hz)
            2 dimension python list, i.e. [[5, 12, 15],[1]], to generate a signal with 2 signals, where the second channel containing 1-Hz signal
        func : string, default: sin
            The periodic function to generate signal, either `sin` or `cos`
        add_noise : float, default: 0
            Add random noise to the signal, where `0` has no noise
        plot : boolean
            Plot the generated signal
    Returns:
        signal : 1d ndarray
            Generated signal, a numpy array of length `sampling_rate*length_seconds`
    """

    frequencies_list = np.array(frequencies_list, dtype=object)
    assert len(frequencies_list.shape) == 1 or len(frequencies_list.shape) == 2, "frequencies_list must be 1d or 2d python list"

    expanded = False
    if isinstance(frequencies_list[0], int):
        frequencies_list = np.expand_dims(frequencies_list, axis=0)
        expanded = True

    npnts = sampling_rate*length_seconds  # number of time samples
    time = np.arange(0, npnts)/sampling_rate
    signal = np.zeros((frequencies_list.shape[0],npnts))

    for channel in range(0,frequencies_list.shape[0]):
        for fi in frequencies_list[channel]:
            if func == "cos":
                signal[channel] = signal[channel] + np.cos(2*np.pi*fi*time)
            else:
                signal[channel] = signal[channel] + np.sin(2*np.pi*fi*time)

        # normalize
        max = np.repeat(signal[channel].max()[np.newaxis], npnts)
        min = np.repeat(signal[channel].min()[np.newaxis], npnts)
        signal[channel] = (2*(signal[channel]-min)/(max-min))-1

    if add_noise:
        noise = np.random.uniform(low=0, high=add_noise, size=(frequencies_list.shape[0],npnts))
        signal = signal + noise

    if plot:
        plt.plot(time, signal.T)
        plt.show()

    if expanded:
        signal = signal[0]

    return signal

#%% LZC function by Schartner
from scipy.signal import hilbert


'''
Python code to compute complexity measures LZc, ACE and SCE as described in "Complexity of multi-dimensional spontaneous EEG decreases during propofol induced general anaesthesia"

Author: m.schartner@sussex.ac.uk
Date: 09.12.14

To compute the complexity meaures LZc, ACE, SCE for continuous multidimensional time series X, where rows are time series (minimum 2), and columns are observations, type the following in ipython:

execfile('CompMeasures.py')
LZc(X)
ACE(X)
SCE(X)


Some functions are shared between the measures.
'''

def Pre(X):
 '''
 Detrend and normalize input data, X a multidimensional time series
 '''
 ro=len(X)
 Z=np.zeros((ro))

 Z=signal.detrend(X-np.mean(X), axis=0)

 return Z


##########
'''
LZc - Lempel-Ziv Complexity, column-by-column concatenation
'''
##########

def cpr(string):
 '''
 Lempel-Ziv-Welch compression of binary input string, e.g. string='0010101'.
 It outputs the size of the dictionary of binary words.
 '''
 d={}
 w = ''
 i=1
 for c in string:
  wc = w + c
  if wc in d:
   w = wc
  else:
   d[wc]=wc
   w = c
  i+=1
 return len(d)

def str_col(X):
 '''
 Input: Continuous multidimensional time series
 Output: One string being the binarized input matrix concatenated comlumn-by-column
 '''
 ro=len(X)
 M=abs(hilbert(X))
 TH=np.mean(M)

 s=''
 for j in range(ro):
   if M[j]>TH:
    s+='1'
   else:
    s+='0'

 return s

def LZc(X):
 '''
 Compute LZc and use shuffled result as normalization
 '''
 X=Pre(X)
 SC=str_col(X)
 M=list(SC)
 np.random.shuffle(M)
 w=''
 for i in range(len(M)):
  w+=M[i]
 return cpr(SC)/float(cpr(w))

def entropy(string):
 '''
 Calculates the Shannon entropy of a string
 '''
 string=list(string)
 prob = [ float(string.count(c)) / len(string) for c in dict.fromkeys(list(string)) ]
 entropy = - sum([ p * np.log(p) / np.log(2.0) for p in prob ])

 return entropy

#%% Generate waveforms
# Sine wave of 10 Hz
sine_wave = generate_signal(30, 250,[10],func="sin", plot=True)
sine_wave_noisy = generate_signal(30, 250,[10],func="sin", plot=True, add_noise = 5)
sine_wave_mix_8_to_12 = generate_signal(30, 250,[8,9,10,11,12],func="sin", plot=True)

# sawtooth
time = np.linspace(0, 30, 250*30)
saw_tooth = signal.sawtooth(2 * np.pi * 5 * time)
plt.plot(time, saw_tooth)

# irregular sawtooth
irregular_sawtooth = np.concatenate(
    [signal.sawtooth(2 * np.pi * np.linspace(0, 1, random.randrange(30, 150))) for _ in range(10)]
)

plt.plot(irregular_sawtooth)

irregular_sawtooths = np.concatenate((irregular_sawtooth,irregular_sawtooth,irregular_sawtooth,
                                      irregular_sawtooth,irregular_sawtooth,irregular_sawtooth,
                                      irregular_sawtooth,irregular_sawtooth,irregular_sawtooth), axis=0)

# gaussian noise
gaussian_noise_series = np.array([gauss(0.0, 1.0) for i in range(7500)])
plt.plot(gaussian_noise_series)

# time series list
wave_names = ['sine_wave','sine_wave_noisy',
           'sine_wave_mix_8_to_12','saw_tooth',
           'irregular_sawtooths','gaussian_noise_series']

parameter_names = ['Permutation entropy',
                   'SVD',
                   'App Entropy',
                   'Sample Entropy',
                   'Hjorth mobility',
                   'Zero crossing',
                   'LZc',
                   'Petrosian FD',
                   'Katz FD',
                   'HFD',
                   'DFA']

wave_list = []
wave_list.append(sine_wave)
wave_list.append(sine_wave_noisy)
wave_list.append(sine_wave_mix_8_to_12)
wave_list.append(saw_tooth)
wave_list.append(irregular_sawtooths)
wave_list.append(gaussian_noise_series)

#%% Complexity analysis
complexity_parameters = []
wave_name = []
parameter_name = []

for wave in range(len(wave_list)):
    complexity = ant.perm_entropy(wave_list[wave], normalize=True) # Permutation entropy
    complexity_parameters.append(complexity)
    wave_name.append(wave_names[wave])
    parameter_name.append(parameter_names[0])

    # Singular value decomposition entropy
    complexity = ant.svd_entropy(wave_list[wave], normalize=True)
    complexity_parameters.append(complexity)
    wave_name.append(wave_names[wave])
    parameter_name.append(parameter_names[1])

    # Approximate entropy
    complexity = ant.app_entropy(wave_list[wave])
    complexity_parameters.append(complexity)
    wave_name.append(wave_names[wave])
    parameter_name.append(parameter_names[2])

    # Sample entropy
    complexity = ant.sample_entropy(wave_list[wave])
    complexity_parameters.append(complexity)
    wave_name.append(wave_names[wave])
    parameter_name.append(parameter_names[3])

    # Hjorth mobility and complexity
    complexity = ant.hjorth_params(wave_list[wave])
    complexity_parameters.append(complexity)
    wave_name.append(wave_names[wave])
    parameter_name.append(parameter_names[4])

    # Number of zero-crossings
    complexity = ant.num_zerocross(wave_list[wave])
    complexity_parameters.append(complexity)
    wave_name.append(wave_names[wave])
    parameter_name.append(parameter_names[5])

    # Lempel-Ziv complexity
    complexity = LZc(wave_list[wave])
    complexity_parameters.append(complexity)
    wave_name.append(wave_names[wave])
    parameter_name.append(parameter_names[6])

    # Petrosian fractal dimension
    complexity = ant.petrosian_fd(wave_list[wave])
    complexity_parameters.append(complexity)
    wave_name.append(wave_names[wave])
    parameter_name.append(parameter_names[7])

    # Katz fractal dimension
    complexity = ant.katz_fd(wave_list[wave])
    complexity_parameters.append(complexity)
    wave_name.append(wave_names[wave])
    parameter_name.append(parameter_names[8])

    # Higuchi fractal dimension
    complexity = ant.higuchi_fd(wave_list[wave])
    complexity_parameters.append(complexity)
    wave_name.append(wave_names[wave])
    parameter_name.append(parameter_names[9])

    # Detrended fluctuation analysis
    complexity = ant.detrended_fluctuation(wave_list[wave])
    complexity_parameters.append(complexity)
    wave_name.append(wave_names[wave])
    parameter_name.append(parameter_names[10])

#%% Generating a dataframe from three lists
dataframe = pd.DataFrame(list(zip(complexity_parameters, parameter_name, wave_name)))
dataframe.columns = ['complexity_parameter','parameter_name','wave_name']

data = dataframe.drop(dataframe.index[dataframe['parameter_name'] == 'Hjorth mobility'],
                      inplace=False)

data.drop(data.index[data['parameter_name'] == 'Zero crossing'],
                      inplace=True)
# to get better visibility by removing Katz parameter
data.drop(data.index[data['parameter_name'] == 'Katz FD'],
                      inplace=True)
#%% Visualisation using seaborn

sns.set(rc={'figure.figsize':(12,10)},
        font_scale = 1)

#deep, pastel, muted, Set2, Paired are some palettes
sns.set_theme(style="whitegrid", palette="deep")

complexity_plot = sns.scatterplot(data = data,
                x="parameter_name",
                y="complexity_parameter",
                hue="wave_name")

complexity_plot.set_xlabel("Complexity parameter", fontsize = 20)
complexity_plot.set_ylabel("Values", fontsize = 20)
complexity_plot.set_title("Complexity parameters for different waveforms",
                          fontsize = 20)

plt.legend(labels=["Sine wave (10 Hz)",
                   "Noisy sine wave",
                   "Sine waves of 8 to 12 Hz mixed",
                   "Saw tooth wave",
                   "Irregular sawtooth wave",
                   "Gaussian noise"])

fig = complexity_plot.get_figure()
fig.savefig(fname = 'comapre_complexity_zoomed.png',
            dpi = 600)

