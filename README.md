# SparkXGBoost 

`SparkXGBoost` aims to implement the [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) tree algorithm in [XGBoost](https://github.com/dmlc/xgboost/) on [Apache Spark](http://spark.apache.org). `SparkXGBoost` is distributed under Apache License 2.0. 

[![Build Status](https://travis-ci.org/rotationsymmetry/SparkXGBoost.svg?branch=master)](https://travis-ci.org/rotationsymmetry/SparkXGBoost) 

## Introduction to Gradient Boosting Trees
The XGBoost team have a fantastic [introduction to gradient boosting trees](http://xgboost.readthedocs.org/en/latest/model.html). Compare with existing which is the basis for the `SparkXGBoost` implementation. 

## Features
`SparkXGBoost` currently supports regression and binary classification using the gradient boosting tree with second order approximation of the loss function. 

* Used defined 1-dimensional loss function
* L2 regularization term on node weights
* Concurrent node learning to 
* Feature sub sampling for learning node

## Components

## Example

## Parameters
The following parameters can be specified by the setters of `SXGBoost` .

* numTrees[default=1]
	* number of trees to be grown in the boosting algorithm.
	* Int, range: [1, ∞]
* gamma [default=0]
	* minimum loss reduction required to make a further partition on a leaf node of the tree. 
	* Double, range: [0,∞]
* maxDepth [default=5]
	* maximum depth of a tree. A tree with one root and two leaves is considered to have depth = 1.
	* Int, range: [1,∞]
* minInstanceWeight [default=1]
	* minimum weight (aka, number of data instance) required to make a further partition on a leaf node of the tree. 
	* Double, range: [0,∞]
* featureSubsampleRatio [default=1]
	* subsample ratio of columns when constructing each tree.
	* Double, range: (0,1]
* lambda [default=0]
	* L2 regularization term on weights. 
	* Double, range: [0,∞]


## Roadmap
I have following tentative roadmap for upcoming releases:

0.2

* Support step size.
* Bagging of the trainning data

0.3

* L1 regularization term on node weights

0.4

* Automatically determine the maximal number of current nodes by memory management.

0.5

* Multi-class classification

0.6 

* Unordered categorical variables

## Contributions
 





