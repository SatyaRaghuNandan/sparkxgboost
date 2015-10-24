# SparkXGBoost 

`SparkXGBoost` aims to implement the [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) tree algorithm in [XGBoost](https://github.com/dmlc/xgboost/) on the [Apache Spark](http://spark.apache.org) platform. `SparkXGBoost` is distributed under Apache License 2.0. 

[![Build Status](https://travis-ci.org/rotationsymmetry/SparkXGBoost.svg?branch=master)](https://travis-ci.org/rotationsymmetry/SparkXGBoost) 
[![codecov.io](https://codecov.io/github/rotationsymmetry/SparkXGBoost/coverage.svg?branch=master)](https://codecov.io/github/rotationsymmetry/SparkXGBoost?branch=master)

## Introduction to Gradient Boosting Trees
The XGBoost team have a fantastic [introduction](http://xgboost.readthedocs.org/en/latest/model.html) to gradient boosting trees, which inspires `SparkXGBoost`. 

## Features
`SparkXGBoost` version 0.1 supports supervised learning using the gradient boosting tree with second order approximation of user-defined loss function. The package comes with `SquareLoss` for regression and `DevianceLoss` for binary classification. 

* User defined 1-dimensional loss function
* L2 regularization term on node weights
* Concurrent learning nodes
* Feature sub sampling for learning nodes

## Components
There are three major components:

`SparkXGBoost` is the learner class. Its constructor takes an instance from the `Loss` class that defines the loss in gradient boosting.  After fitting the tree ensembles with the training data, it will produce the model as an instance from `SparkXGBoost` class. 

``` scala
class SparkXGBoost(val loss: Loss){
  def fit(dataset: DataFrame): SparkXGBoostModel
}
```

`SparkXGBoost` contains the trained tree ensemble and is capable to making predictions for the instances.

``` scala
class SparkXGBoostModel {
  // Predict label given the feature of a single instance
  def predict(features: Vector): Double
  // Provide prediction for the entire dataset
  def transform(dataset: DataFrame): SparkXGBoostModel
}
```

The abstract class `Loss` defines the contract for user-defined loss functions. 

``` scala
abstract class Loss{
  // The 1st derivative
  def diff1(label: Double, f: Double): Double
  // The 2nd derivative 
  def diff2(label: Double, f: Double): Double
  // Generate prediction from the score suggested by the tree ensemble
  def toPrediction(score: Double): Double
  // obtain bias 
  def getInitialBias(input: RDD[LabeledPoint]): Double
}
```
  
## Example

## Parameters
The following parameters can be specified by the setters in `SXGBoost` .

* labelCol[default="label"]
	* the name of the label column of the `DataFrame`
	* String
* featuresCol[default="features"]
	* the name of the feature column of the `DataFrame`
	* String
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


The following parameters can be specified by the setters in `SXGBoostModel` .

* predictionCol[default="prediction"]
	* the name of the prediction column of the `DataFrame`
	* String
* featuresCol[default="features"]
	* the name of the feature column of the  `DataFrame`
	* String

## Roadmap
I have following tentative roadmap for the upcoming releases:

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
 
Many thanks for testing `SparkXGBoost`. You can file bug report or provide suggestions using [github issues](https://github.com/rotationsymmetry/SparkXGBoost/issues). If you would like to improve the codebase, please don't hesitate to submit a pull request. 
