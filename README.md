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

![](https://github.com/rahulvenugopal/ComapreComplexity/blob/main/images/comapre_complexity.png)

- Zooming in by removing Katz dimension for scaling
![](https://github.com/rahulvenugopal/ComapreComplexity/blob/main/images/comapre_complexity_zoomed.png)