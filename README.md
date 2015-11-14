# SparkXGBoost 

SparkXGBoost is a [Spark](http://spark.apache.org) implementation of [gradient boosting tree](https://en.wikipedia.org/wiki/Gradient_boosting) using 2nd order approximation 
of arbitrary user-defined loss function. SparkXGBoost is inspired by the [XGBoost](https://github.com/dmlc/xgboost/) project.

`SparkXGBoost` is distributed under Apache License 2.0. 

[![Build Status](https://travis-ci.org/rotationsymmetry/sparkxgboost.svg?branch=master)](https://travis-ci.org/rotationsymmetry/sparkxgboost)
[![codecov.io](https://codecov.io/github/rotationsymmetry/sparkxgboost/coverage.svg?branch=master)](https://codecov.io/github/rotationsymmetry/sparkxgboost?branch=master)

## What is Gradient Boosting Tree?
The XGBoost team have a fantastic [introduction](http://xgboost.readthedocs.org/en/latest/model.html) to gradient boosting trees. 

## Features
SparkXGBoost version supports supervised learning with the gradient boosting tree using 2nd order approximation of arbitrary user-defined loss function. SparkXGBoost ships with The following `Loss` classes: 

* `SquareLoss` for linear (normal) regression
* `LogisticLoss` for binary classification
* `PoissonLoss` for Poisson regression of count data

To avoid overfitting, SparkXGBoost employs the following regularization methods: 

* Shrinkage by learning rate (aka step size)
* L2 regularization term on node
* L1 regularization term on node
* Stochastic gradient boosting (similar to Bagging)
* Feature sub sampling for learning nodes

SparkXGBoost is capable of processing multiple learning nodes in the one pass of the training data to improve efficiency. 

## Design

SparkXGBoost implements the Spark ML [Pipeline](http://spark.apache.org/docs/latest/ml-guide.html#pipeline) API, allowing
you to easily run a sequence of algorithms to process and learn from data.

* `SparkXGBoostRegressor` and `SparkXGBoostRegressionModel` are the predictor and model for continuous labels.
* `SparkXGBoostClassifier` and `SparkXGBoostClassificationModel` are the predictor and model for categorical labels.

In the constructors of `SparkXGBoostRegressor` and `SparkXGBoostClassifier`, users will need to supply an instance of
the `Loss` class to define the loss functions and its derivatives. SparkXGBoost currently comes with
`SquareLoss` for linear (normal) regression, `LogisticLoss` for binary classification and
`PoissonLoss` for Poisson regression of count data. Additional loss function can be specified by the user
by sub-classing the `Loss`.

``` scala
abstract class Loss{
  // The 1st derivative
  def diff1(label: Double, f: Double): Double
  // The 2nd derivative 
  def diff2(label: Double, f: Double): Double
  // Generate prediction from the score suggested by the tree ensemble
  // For regression, prediction is the label
  // For classification, prediction is the probability in each class
  def toPrediction(score: Double): Double
  // Calculate bias 
  def getInitialBias(input: RDD[LabeledPoint]): Double
}
```

Please see the example below for typical usage.

## Example

`trainingData` is a `DataFrame` with the labels stored in a column named "label" and the feature vectors stored in a column name "features".  Similarly, `testData` is `DataFrame` with the feature vectors stored in a column name "features".

Please note that the feature vectors have to been indexed before feeding to the pipeline to ensure the categorical variables are correctly encoded with metadata.

Currently, all categorical variables are assumed to be ordered. Unordered categorical variables can be used for training after being coded with [OneHotEncoder](http://spark.apache.org/docs/latest/ml-features.html#onehotencoder).

``` scala
  val featureIndexer = new VectorIndexer()
    .setInputCol("features")
    .setOutputCol("indexedFeatures")
    .setMaxCategories(2)
    .fit(trainingData)

  val sparkXGBoostRegressor = new SparkXGBoostRegressor(new SquareLoss)
    .setFeaturesCol("indexedFeatures")
    .setMaxDepth(2)
    .setNumTrees(5)

  val pipeline = new Pipeline()
    .setStages(Array(featureIndexer, sparkXGBoostRegressor))

  val model = pipeline.fit(data)

  val prediction = model.transform(testData)
```

## Parameters
The following parameters can be specified by the setters.

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
    * learning rate (aka step size) for gradient boosting.
    * Double, range: (0, 1]
* minInstanceWeight [default=1]
	* minimum weight (aka, number of data instance) required to make a further partition on a leaf node of the tree.
	* Double, range: [0, ∞]
* sampleRatio [default=1.0]
    * sample ratio of rows in bagging
    * Double, range(0, 1]
* featureSampleRatio [default=1.0]
	* sample ratio of columns when constructing each tree.
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

## Compatibility 
SparkXGBoost has been tested with Spark 1.5.1 and Scala 2.10.

## Use SparkXGBoost in Your Project

### Option 1: spark-package.org
Releases of SparkXGBoost are available on [spark-package.org](http://spark-packages.org/package/rotationsymmetry/sparkxgboost). 
You can follow the "How to" for spark-shell, sbt or maven.

As SparkXGBoost is currently under active development, 
the spark-package.org release might not always include the latest update.

### Option 2: Compile 
You can access the latest cutting edge codebase through compilation from the source.

Step 1: clone the project from GitHub

``` bash
git clone https://github.com/rotationsymmetry/sparkxgboost.git
```

Step 2: compile and package the jar using [sbt](http://www.scala-sbt.org)

``` bash 
cd SparkXGBoost
sbt clean package
```

You should be able to find the jar file in `target/target/scala-2.10/sparkxgboost_2.10-x.y.z.jar`

Step 3: load it in your Spark project

* If you are using spark-shell, you can type in

``` bash
./spark-shell --jars path/to/sparkxgboost_2.10-x.y.z.jar
```

* If you are building Spark application with sbt, you can put the jar file into the `lib` folder next to `src`. Then sbt should be able to put SparkXGBoost in your class path.


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
