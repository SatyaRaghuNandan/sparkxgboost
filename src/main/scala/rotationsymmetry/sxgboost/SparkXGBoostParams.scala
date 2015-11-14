package rotationsymmetry.sxgboost

import org.apache.spark.ml.param._

trait SparkXGBoostParams extends Params {

  final val alpha: DoubleParam = new DoubleParam(this, "alpha",
    "L1 regularization term on weights")
  setDefault(alpha -> 0.0)

  final val lambda: DoubleParam = new DoubleParam(this, "lambda",
    "L2 regularization term on weights")
  setDefault(lambda -> 0.0)

  final val eta: DoubleParam = new DoubleParam(this, "eta",
    "learning rate (aka step size) for gradient boosting")
  setDefault(eta -> 1.0)

  final val gamma: DoubleParam = new DoubleParam(this, "gamma",
    "minimum loss reduction required to make a further partition on a leaf node of the tree")
  setDefault(gamma -> 0.0)

  final val numTrees: IntParam = new IntParam(this, "numTrees",
    "number of trees to be grown in the boosting algorithm")
  setDefault(numTrees -> 1)

  final val maxDepth: IntParam = new IntParam(this, "maxDepth",
    "maximum depth of a tree. A tree with one root and two leaves is considered to have depth = 1")
  setDefault(maxDepth -> 5)

  final val minInstanceWeight: DoubleParam = new DoubleParam(this, "minInstanceWeight",
    "minimum weight (aka, number of data instance) required to make a further partition on a leaf node of the tree")
  setDefault(minInstanceWeight -> 1.0)

  final val sampleRatio: DoubleParam = new DoubleParam(this, "sampleRatio",
    "sample ratio of rows in bagging")
  setDefault(sampleRatio -> 1.0)

  final val featureSampleRatio: DoubleParam = new DoubleParam(this, "featureSampleRatio",
    "sample ratio of columns when constructing each tree")
  setDefault(featureSampleRatio -> 1.0)

  final val maxConcurrentNodes: IntParam = new IntParam(this, "maxConcurrentNodes",
    "maximal number of nodes to be process in one pass of the training data")
  setDefault(maxConcurrentNodes -> 50)

  final val maxBins: IntParam = new IntParam(this, "maxBins",
    "maximal number of bins for continuous variables")
  setDefault(maxBins -> 32)

  final val seed: LongParam = new LongParam(this, "seed",
    "random seed")
  setDefault(seed, 1L)
}
