# SparkXGBoost 

SparkXGBoost aims to implement the [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting) tree algorithm in [XGBoost](https://github.com/dmlc/xgboost/) on the [Apache Spark](http://spark.apache.org) platform. `SparkXGBoost` is distributed under Apache License 2.0. 

[![Build Status](https://travis-ci.org/rotationsymmetry/SparkXGBoost.svg?branch=master)](https://travis-ci.org/rotationsymmetry/SparkXGBoost) 
[![codecov.io](https://codecov.io/github/rotationsymmetry/SparkXGBoost/coverage.svg?branch=master)](https://codecov.io/github/rotationsymmetry/SparkXGBoost?branch=master)

## Introduction to Gradient Boosting Trees
The XGBoost team have a fantastic [introduction](http://xgboost.readthedocs.org/en/latest/model.html) to gradient boosting trees, which inspires `SparkXGBoost`. 

## Features
SparkXGBoost version 0.1 supports supervised learning using the gradient boosting tree with second order approximation of arbitrary user-defined loss function. The following `Loss` classes are provided with the package: 

* `SquareLoss` for linear (normal) regression
* `LogisticLoss` for binary classification
* `PoissonLoss` for Poisson regression of count data

SparkXGBoost includes following approach to avoid overfitting

* L2 regularization term on node
* L1 regularization term on node
* Feature sub sampling for learning nodes

SparkXGBoost is capable of processing multiple learning nodes in the one pass of the training data to improve efficiency. 

## Components
There are three major components:

`SparkXGBoost` is the learner class. Its constructor takes an instance from the `Loss` class that defines the loss in gradient boosting.  After fitting the tree ensembles with the training data, it will produce the model as an instance from `SparkXGBoost` class. 

``` scala
class SparkXGBoost(val loss: Loss){
  def fit(dataset: DataFrame): SparkXGBoostModel
}
```

`SparkXGBoostModel` contains the trained tree ensemble and is capable to making predictions for the instances.

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
  
## Usage Guide and Example

Below is an example running SparkXGBoost. `trainingData` is a `DataFrame` with the labels stored in a column named "label" and the feature vectors stored in a column name "features".  Similarly, `testData` is `DataFrame` with the feature vectors stored in a column name "features". 

Pleaes note that the feature vectors have to been indexed before feeding to the `SparkXGBoost` and `SparkXGBoostModel` to ensure the categorical variables are correctly encoded with metadata.

In SparkXGBoost 0.1, all categorical variables are assumed to be ordered. Unordered categorical variables can be used for training after being coded with [OneHotEncoder](http://spark.apache.org/docs/latest/ml-features.html#onehotencoder). 

``` scala
  val featureIndexer = new VectorIndexer()
    .setInputCol("features")
    .setOutputCol("indexedFeatures")
    .setMaxCategories(2)
    .fit(trainingData)

  val sXGBoost = new SparkXGBoost(new SquareLoss)
    .setFeaturesCol("indexedFeatures")
    .setMaxDepth(1)
    .setNumTrees(1)
  val sXGBoostModel = sXGBoost.fit(
    featureIndexer.transform(trainingData))

  val predictionData = sXGBoostModel.transform(
    featureIndexer.transform(testData))
```

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
* alpha [default=0]
	* L1 regularization term on weights. 
	* Double, range: [0,∞]
* maxConcurrentNodes[default=50]
	* maximal number of nodes to be process in one pass of the training data.

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

0.3

* Post-pruning

0.4

* Automatically determine the maximal number of current nodes by memory management.

0.5

* Multi-class classification

0.6 

* Unordered categorical variables

## Bugs and Improvements
 
Many thanks for testing SparkXGBoost! 

You can file bug report or provide suggestions using [github issues](https://github.com/rotationsymmetry/SparkXGBoost/issues). 

If you would like to improve the codebase, please don't hesitate to submit a pull request. 
