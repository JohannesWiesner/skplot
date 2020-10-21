# skplot

A python package for extracting, plotting and reporting information from one or multiple sklearn classification & prediction pipelines.

## Features

skplot contains four modules that facilitate the final goal of data visualization.

- extraction.py (helps with data extraction from sklearn outputs)
- preparation.py (helps to prepare the extracted data in order to plot it)
- plotting.py (plot one or multiple outputs from sklearn)
- report.py (create statistical reports)

## Example

See examples/breast_cancer.py for a small introduction on how to use skplot. skplot is able to produce outputs like in the plot below that visualizes how different
input parameters of your pipeline affect your training and test scores:

You can plot it like this:

![Example 1](/examples/breast_cancer_catplot.png)

Or like this, depending on your taste:

![Example 2](/examples/breast_cancer_lineplot.png)

