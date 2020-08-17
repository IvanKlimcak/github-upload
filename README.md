# scorecard_dev

This packages serves as utility box for model development. It follows general stucture of model development process.
The goal is to make a traditional scorecard using logistic regression.

The package contains functions to tackle all general areas of model development:
- Explanatory data analysis
- Data profiling (missing values and outliers)
- Predictive power measurements 
- Sampling and binning 
- Coarse classing and WoE
- Logistic regression
- Performance evaluation
- Scorecard scaling and deployment

## Installation
- Currently the package is not released

## Explanatory data analysis and data profiling
- `univariate_analysis`, provides basic location and dispersion measures for a given variable 
- `plot_cat_var`, plots all categorical variables 
- `missing_values`, calculates proportion of missing values for all variables 
- `identical_values`, calculates proportion of identical values for all variables
- `outliers_detection`, serves as identification of outliers

## Predictive power measurements
- `iv_apply`, calcualates information value for all variables
- `correlation_measures` calculates pearson correlation measures for all variables

# Sampling 
- `split_dataset` conditionally or unconditionally split dataset 
