# mw_density_imig2025
This repository contains the code and results from [Imig et al. 2025](https://astrojimig.github.io/pdfs/Imig_MW_density.pdf)

If you have any questions, please contact the author: Julie Imig (jimig@stsci.edu)

### <font color='red'>!!! This repository is still a work in progress !!! </font>

# Overview

## The Jupyter Notebooks

The code for reproducing the results and figures from [Imig et al. 2025](https://astrojimig.github.io/pdfs/Imig_MW_density.pdf) is set up in a series of jupyter notebooks in this directory. A brief overiew of each of them is provided below:

### Part 0: Sample Selection

The notebook `Part0_Sample_Selection.ipynb`...

### Part 1: Selection Function
The notebook `Part1_Selection Function.ipynb`...

### Part 2: Density Fits
The notebook `Part2_Density_Fits.ipynb`...

### Part 3: Plots and Analysis
The notebook `Part3_Plot_and_Analysis.ipynb`...


## Other Directories
### mw_density
The `mw_density/` directory contains most of the source code for this pacakge. The...

### Data
The `data/` directory contains...

### Results
The `results/` directory contains...



## Installation Instructions

First, download this repository by cloning it with:

```
git clone https://github.com/astrojimig/mw_density_imig2025
```

Set up a new conda environment named "mw_density":
```
conda create -n mw_density python==3.11
```

Activate your conda environment:
```
conda activate mw_density
```

Finally, install this code and all of the relevant dependencies with:

```
pip install -e .
```

The `-e` flag installs the repository in *editable* mode, meaning any changes you make to the files will automatically be saved and you won't need to install it again.


## Authors
- Julie Imig, Space Telescope Science Institute (jimig@stsci.edu)

Please reach out if you have any questions or ideas!