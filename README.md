# Complexity analysis in neuroscience
![muppets](https://raw.githubusercontent.com/rahulvenugopal/ComapreComplexity/main/images/muppet_show.png)

- Just like the above image, there are so many measures/methods available in complexity analysis
- Which one to pick is tricky
- On top of it, each method comes with its own hyper parameters (we have to decide optimal values for these hyper parametres)
- How do we tune them for say EEG data vs RR interavls from ECG data vs tremor data from an accelerometer!
- What all assumptions should be met by the data before appying these methods?
- How do we interpret these parameters?
- What are the relationships between these parameters?

---
## Compare various complexity measures
- Six different waveforms were created with same data lengths
- `Nine` complexity methods were computed on these datasets
- Default parameters were applied (**no** hyper tuning of parameters)
- Schartner's code for calcualting Lempel Ziv complexity was modified for 1D data. See the reference [paper](https://pubmed.ncbi.nlm.nih.gov/26252378/)
- Antropy module's LZc code was extremely slow(>30 minutes for a 7500 data points long time series)

---
- Here are the images from comparison table

![cc](https://github.com/rahulvenugopal/ComapreComplexity/blob/main/images/comapre_complexity.png)

- Zooming in by removing Katz dimension for scaling
![zoom](https://github.com/rahulvenugopal/ComapreComplexity/blob/main/images/comapre_complexity_zoomed.png)

- Refer below image to see the waveforms
![wave_collage](https://raw.githubusercontent.com/rahulvenugopal/ComapreComplexity/main/images/FinalCollage.png)

- The deep question is what aspects of complexity, regularity, patterns are geting captured in each method

---
## Detrended Fluctuation Analysis (DFA) | Based on the awesome nold package's [source code[(https://nolds.readthedocs.io/en/latest/_modules/nolds/measures.html#dfa)
![DFA](https://github.com/rahulvenugopal/ComapreComplexity/blob/main/images/DFA.jpg)

- Look at the colored wiggles and lines respectively from B and C
- The lines capture the trends which control the wiggles
- My understanding is that, these trends prevent us from picking up patterns
- So detrend and hunt for patterns. Will cite reference so that we can tumble down that rabbit hole :rabbit:

### Part 1
- Start with 1D time series data
- Subtract mean from the entire data
- Now do a cumulative sum of these deviations from mean
- That's the data which goes to DFA analysis. Remember, this is still a 1D time series

### Part 2
- Create different window lengths (data pieces which are 10 datapoints, 20, 100 etc. etc.)
- This starts with just `4` data points long window upto window sizes `1/10th` of total data lengths
- The window lengths are set at logarithmic steps

### Part 3
- For example, take a window length of 14 datapoints. If our 1D series is 1000 datapoints long, we can get `141` pieces (with 50% data overlap)
- We get a 2D array now, where we have 141 rows and each row has a data which is of length 14
- For each of these rows, fit a line and get parameters of fit (intercept and slope)
- We will get 141 slopes and intercepts for window length `14`
- For each window length, repeat the same. Fit a line and save the parametres of line fit (slope and intercept)

### Part 4
- Once the slope and intercept is estimated, we know the trend (as a line)
- Detrend the data (data - line) using the trend line (as per previous fit) and find standard deviation of this time series
- Find the average of all these fluctuations across various segments
- For one window length we get `one` average `standard deviation` value
- After doing above steps we get `SD` values for each `window length`
- Finally, plot these fluctuations and window lengths on a `log-log` scale
- Fit a robust line (using RANSAC method, robust to outliers)
- DFA exponent is the slope of this robust fit

#### To be updated
- Explanations, interpretations, intuitions (why cumulative sum, why detrending etc. etc.)
- I am open to collaborations on these aspects and beyond :handshake:

---
## Sample Entropy | Based on the awesome Antropy package's [source code](https://github.com/raphaelvallat/antropy/blob/master/antropy/entropy.py)


---
## Permutation entropy
- Coming next