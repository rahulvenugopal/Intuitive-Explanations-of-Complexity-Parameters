# Complexity analysis in neuroscience
![muppets](https://raw.githubusercontent.com/rahulvenugopal/ComapreComplexity/main/images/muppet_show.png)

- Just like the above image, there are so many measures/methods available in complexity analysis
- Which one to pick is tricky
- On top of it, each method comes with its own hyper parameters
- How do we tune them for say EEG data!
- What all assumptions should be met by the data before appying these methods?
- How do we interpret these parameters?
- What are the relationshiups between these parameters

---
- Six different waveforms were created with same lengths
- Nine complexity methods were computed on these datasets
- Default parameters were applied (no hyper tuning of parameters)
- Schartner's code for calcualting Lempel Ziv complexity was modified for 1D data
- Antropy module's LZc code was extremely slow(>30 minutes for a 7500 data points long time series)

---
- Here are the images from comparison table

![cc](https://github.com/rahulvenugopal/ComapreComplexity/blob/main/images/comapre_complexity.png)

- Zooming in by removing Katz dimension for scaling
![zoom](https://github.com/rahulvenugopal/ComapreComplexity/blob/main/images/comapre_complexity_zoomed.png)

- Refer below image to see the waveforms
![wave_collage](https://raw.githubusercontent.com/rahulvenugopal/ComapreComplexity/main/images/FinalCollage.png)

---
## Detrended Fluctuation Analysis (DFA)
![DFA](https://github.com/rahulvenugopal/ComapreComplexity/blob/main/images/DFA.jpg)

## Part 1
- start with 1D time series data
- create different window lengths
- this starts with just `4` data points long window upto window sizes `1/10th` of total data lengths
- The window lengths are set at logarithmic gaps
- subtract mean from the entire data
- now do a cimulative sum of these deviations from mean
- that's the data which goes to DFA analysis

### Part 2
- for each window length, fit a line and save the parametres of line fit (slope and intercept)
- within each window length, this fitting of a line is done in a sliding windiw manner
- detrend the data using the trend line (as per previous fit) and find standard deviation of this time series
- find the average of all these fluctuations across various segments
- for one window length we get `one` average `standard deviation` value
- after doing above steps we get different `SD` values for different `window lengths`
- finally, plot these fluctuations and window lengths on a log-log scale
- fir a robust line (using RANSAC method)
- DFA is the parameters of this robust fit

## To Do
- Explanations, interpretations