# mw_density_imig2025
This repository contains the results, source code and data used in [Imig et al. 2025](https://astrojimig.github.io/pdfs/Imig_MW_density.pdf) for studying the density structure of the Milky Way disk.

<p>
<center>
<button name="button" onclick="https://astrojimig.github.io/pdfs/Imig_MW_density.pdf">Read the paper here!</button>
</center>
</p>

If you have any questions or ideas, please contact the author: Julie Imig (jimig@stsci.edu)

### <font color='red'>!!! This repository is still a work in progress !!! The code is still subject to edits until the paper is published. </font>

# Introduction

Introduction paragraph here...


Scroll down to the [Installation Instructions](#installation-instructions) for guidance on how to clone and install this code.

# Code Overview

## The Jupyter Notebooks

The main code for reproducing the results and figures from [Imig et al. 2025](https://astrojimig.github.io/pdfs/Imig_MW_density.pdf) is set up in a series of Jupyter notebooks in this directory. A brief overiew of each of them is provided below:

### Part 0: Sample Selection

Notebook: [`Part0_Sample_Selection.ipynb`](Part0_Sample_Selection.ipynb)|


This notebook defines the data sample from the the APOGEE `allStar` file and produces plots showing various overviews and summaries of the data sample and of each defined stellar population. This includes Figures *X-Y* from the paper.

### Part 1: Selection Function

Notebook: [`Part1_Selection_Function.ipynb`](Part1_Selection_Function.ipynb)

This notebook generates the raw and effective Selection Functions for APOGEE DR17 using the [`apogee` package](https://github.com/astrojimig/apogee/tree/dr17-selection), originally developed in [Bovy et al. 2012](http://arxiv.org/abs/1510.06745) and updated for DR17 in this work.

This notebook... and includes Figures *X-Y* from the paper.

### Part 2: Density Fits
Notebook: [`Part2_Density_Fits.ipynb`](`Part2_Density_Fits.ipynb)

This notebook contains the code necessary for running the MCMC fits for finding the best-fit density parameters for each stellar population.

### Part 3: Plots and Analysis
Notebook: [`Part3_Plot_and_Analysis.ipynb`](Part3_Plot_and_Analysis.ipynb)

The final notebook in this repository analyzes the results from the MCMC fits and creates the remainder of the figures from the paper. This includes...

## Results & Data

If you're interested in downloading the results of this study in a table format, look no further! All of the...

<p>
<center>
<button name="button" onclick="">Download density_params.fits</button>
</center>
</p>

### Results
The `results/` directory contains the main results and all of the figures for the paper [Imig et al. 2025](https://astrojimig.github.io/pdfs/Imig_MW_density.pdf). 


### Source Data
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

To install this code, first clone this repository into a location of your choice:

```
git clone https://github.com/astrojimig/mw_density_imig2025
```

 Next, set up a new conda environment named "mw_density":

```
conda create -n mw_density python==3.11
```

Activate your conda environment with:
```
conda activate mw_density
```

Next, install this code and all of the relevant dependencies with:

```
pip install -e .
```

The `-e` flag installs the repository in *editable* mode, meaning any changes you make to the files will automatically be saved and you won't need to install it again.

Finally, launch a Jupyter Notebook kernel to view and run the notebooks described above:
```
jupyter notebook
```

# Citation
This work is published under a [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) creative commons liscense.

If you use any part of this code in any published work, please make sure to cite the following papers:
- [Imig et al. 2025](https://astrojimig.github.io/pdfs/Imig_MW_density.pdf)

# Authors
- Julie Imig, Space Telescope Science Institute (jimig@stsci.edu)

Please reach out if you have any questions or ideas!