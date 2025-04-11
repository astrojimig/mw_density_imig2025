# mw_density_imig2025
This repository contains the code and results from [Imig et al. 2025](https://astrojimig.github.io/pdfs/Imig_MW_density.pdf)

If you have any questions, please contact the author: Julie Imig (jimig@stsci.edu)

### <font color='red'>!!! This repository is still a work in progress !!! </font>

# Introduction

Introduction paragraph here...


# The Jupyter Notebooks

The code for reproducing the results and figures from [Imig et al. 2025](https://astrojimig.github.io/pdfs/Imig_MW_density.pdf) is set up in a series of Jupyter notebooks in this directory. A brief overiew of each of them is provided below:

### Part 0: Sample Selection

The notebook `Part0_Sample_Selection.ipynb`...

### Part 1: Selection Function
The notebook `Part1_Selection Function.ipynb`...

### Part 2: Density Fits
The notebook `Part2_Density_Fits.ipynb`...

### Part 3: Plots and Analysis
The notebook `Part3_Plot_and_Analysis.ipynb`...


# Results & Data

## Results
The `results/` directory contains the main results and all of the figures for the paper [Imig et al. 2025](https://astrojimig.github.io/pdfs/Imig_MW_density.pdf). 



## Data
The `data/` directory contains directory contains copies of the ancillary source data used in this study. This includes:

- [`isochrones/parsec.dat`](data/isochrones/parsec.dat): a set of [PARSEC isochrones](https://stev.oapd.inaf.it/cgi-bin/cmd) for the age and metallicity limits of the sample ([Bressan et al, 2012](http://dx.doi.org/10.1111/j.1365-2966.2012.21948.x), [Chen et al. 2014](https://ui.adsabs.harvard.edu/abs/2014MNRAS.444.2525C))
- [`manga_mwas/`](data/manga_mwas) The sample of Milky Way Analog (MWA) galaxies from the MaNGA survey used in [Boardman et al. 2020](https://doi.org/10.1093%2Fmnras%2Fstaa2731), including selection criteria and combined spectra.
- [`selfuncs/`]() - the raw selection function for APOGEE DR17 calculated as part of this study ([Imig et al. 2025](https://astrojimig.github.io/pdfs/Imig_MW_density.pdf))
- [`sloan_filters/filter_curves.fits`](data/sloan_filters/filter_curves.fits): The filter curves for the SDSS *ugriz* filters [(Gunn et al. 1998)](http://adsabs.harvard.edu/abs/1998AJ....116.3040G)
- [`ssps/`](data/ssps/) : The MaStar Simple Stellar Population models from [Maraston et al. 2020](https://doi.org/10.1093%2Fmnras%2Fstaa1489)
- [`apogee_sample.fits`]()

## Source Code
The `mw_density/` directory contains most of the source code for this package. Each...

A brief summary of each file:
- [`density_profiles.py`](mw_density/density_profiles.py): Various equations for defining the parameterized density models
- [`mcmc_functions.py`](mw_density/mcmc_functions.py): Setup and functions for performing the MCMC fits
- [`plotting_helpers.py`](mw_density/plotting_helpers.py): Various functions to help make plots
- [`sample_selection.py`](mw_density/sample_selection.py): Definitions and scripts for refining the data sample from the `allStar` and `DistMass` files
- [`selection_function.py`](mw_density/selection_function.py): A wrapper class for the APOGEE DR17 selection function


## Installation Instructions

To install this code...
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


# Authors
- Julie Imig, Space Telescope Science Institute (jimig@stsci.edu)

Please reach out if you have any questions or ideas!