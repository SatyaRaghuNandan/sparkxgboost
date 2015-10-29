# SparkXGBoost 

SparkXGBoost is a [Spark](http://spark.apache.org) implementation of [gradient boosting tree](https://en.wikipedia.org/wiki/Gradient_boosting) using 2nd order approximation 
of arbitrary user-defined loss function. SparkXGBoost is inspired by the [XGBoost](https://github.com/dmlc/xgboost/) project.

`SparkXGBoost` is distributed under Apache License 2.0. 

[![Build Status](https://travis-ci.org/rotationsymmetry/SparkXGBoost.svg?branch=master)](https://travis-ci.org/rotationsymmetry/SparkXGBoost) 
[![codecov.io](https://codecov.io/github/rotationsymmetry/SparkXGBoost/coverage.svg?branch=master)](https://codecov.io/github/rotationsymmetry/SparkXGBoost?branch=master)

## What is Gradient Boosting Tree?
The XGBoost team have a fantastic [introduction](http://xgboost.readthedocs.org/en/latest/model.html) to gradient boosting trees. 

## Features
SparkXGBoost version supports supervised learning with the gradient boosting tree using 2nd order approximation of arbitrary user-defined loss function. SparkXGBoost ships with The following `Loss` classes: 

* `SquareLoss` for linear (normal) regression
* `LogisticLoss` for binary classification
* `PoissonLoss` for Poisson regression of count data

To avoid overfitting, SparkXGBoost employs the following regularization methods: 

* L2 regularization term on node
* L1 regularization term on node
* Stochastic gradient boosting (similar to Bagging)
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
  // Calculate bias 
  def getInitialBias(input: RDD[LabeledPoint]): Double
}
```

## Compatibility 
SparkXGBoost has been tested with Spark 1.5.1/1.4.1 and Scala 2.10.

## Use SparkXGBoost in Your Project

Firstly, clone the project from GitHub

``` bash
git clone https://github.com/rotationsymmetry/SparkXGBoost.git
```

Secondly, compile and package the jar using [sbt](http://www.scala-sbt.org)

``` bash 
cd SparkXGBoost
sbt package clean package
```

You should be able to find the jar file in `target/target/scala-2.10/sparkxgboost_2.10-0.2.jar`

Lastly, load it in your Spark project

* If you are using spark-shell, you can type in

``` bash
./spark-shell --jars path/to/sparkxgboost_2.10-0.2.jar
```

* If you are building Spark application with sbt, you can put the jar file into the `lib` folder next to `src`. Then sbt should be able to put SparkXGBoost in your class path.

## Example

Below is an example running SparkXGBoost. `trainingData` is a `DataFrame` with the labels stored in a column named "label" and the feature vectors stored in a column name "features".  Similarly, `testData` is `DataFrame` with the feature vectors stored in a column name "features". 

Pleaes note that the feature vectors have to been indexed before feeding to the `SparkXGBoost` and `SparkXGBoostModel` to ensure the categorical variables are correctly encoded with metadata.

Currently, all categorical variables are assumed to be ordered. Unordered categorical variables can be used for training after being coded with [OneHotEncoder](http://spark.apache.org/docs/latest/ml-features.html#onehotencoder). 

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

* labelCol [default="label"]
	* the name of the label column of the `DataFrame`
	* String
* featuresCol [default="features"]
	* the name of the feature column of the `DataFrame`
	* String
* numTrees [default=1]
	* number of trees to be grown in the boosting algorithm.
	* Int, range: [1, ∞]
* maxDepth [default=5]
	* maximum depth of a tree. A tree with one root and two leaves is considered to have depth = 1.
	* Int, range: [1,∞]
* lambda [default=0]
	* L2 regularization term on weights. 
	* Double, range: [0, ∞]
* alpha [default=0]
	* L1 regularization term on weights. 
	* Double, range: [0, ∞]
* gamma [default=0]
	* minimum loss reduction required to make a further partition on a leaf node of the tree. 
	* Double, range: [0, ∞]
* eta [default=1.0]
    * learning rate, or step size for gradient boosting. 
    * Double, range: (0, 1]
* minInstanceWeight [default=1]
	* minimum weight (aka, number of data instance) required to make a further partition on a leaf node of the tree. 
	* Double, range: [0, ∞]
* sampleRatio [default=1.0]
    * sample ratio of rows in bagging
    * Double, range(0, 1]
* featureSubsampleRatio [default=1.0]
	* subsample ratio of columns when constructing each tree.
	* Double, range: (0, 1]
* maxConcurrentNodes [default=50]
	* maximal number of nodes to be process in one pass of the training data.
	* Int, [1, ∞]
* maxBins [default=32]
    * maximal number of bins for continuous variables.
    * Int, [2, ∞]
* seed [default = some random value]
    * seed of sampling.
    * Long
    
The following parameters can be specified by the setters in `SXGBoostModel` .

* predictionCol [default="prediction"]
	* the name of the prediction column of the `DataFrame`
	* String
* featuresCol [default="features"]
	* the name of the feature column of the  `DataFrame`
	* String

## Roadmap
I have following tentative roadmap for the upcoming releases:

0.3

* Post-pruning

0.4

* Automatically determine the maximal number of current nodes by memory management

0.5

* Multi-class classification

0.6 

* Unordered categorical variables

## Bugs and Improvements
 
Many thanks for testing SparkXGBoost! 

You can file bug report or provide suggestions using [GitHub Issues](https://github.com/rotationsymmetry/SparkXGBoost/issues). 

If you would like to improve the codebase, please don't hesitate to submit a pull request. 
