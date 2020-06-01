# TDACrimeRatePred
Uses Topological Data Analysis to analyze Boston Crime Data and Predict Crime rates in each district by month

## Dependencies
1.  numpy
2.  ripser
3.  persim
4.  pandas
5.  time
6.  matplotlib
7.  seaborn
8.  sklearn

All dependencies can be installed via the terminal using the command:
```bash
pip3 install [module name]
```

## complexity_analysis.py
plots out the time complexity of the veitrous rips algorithm

Can be run in the terminal using the command:
```bash
python3 complexity_analysis.py
```

## district_risper.py
Uses the veitrous rips algorithm to compute the persistence homology of the data and
all of the preprocessing needed for the data to be used in the regression models

It outputs the training and testing sets to be used in the regression model as csv files

The code can be run in the terminal using the command:
```bash
python3 district_ripser.py
```

## crime_regression.py
Generates the regression models and plots the predicted values from the regression
models against the actual values. It also outputs the error for each prediction from each regression
model as well as the average error for the entire regression model

The code can be run in the terminal using the command:
```bash
python3 crime_regression.py
```

*district_ripser.py must be run before crime_regression.py is run

## Crimes-in-boston.zip
zip file containing the Boston crime data
file must be extracted from the zip and placed in the same directory as the code being run
