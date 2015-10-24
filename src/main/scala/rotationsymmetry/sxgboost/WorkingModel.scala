package rotationsymmetry.sxgboost

import rotationsymmetry.sxgboost.loss.Loss

class WorkingModel(bias: Double, var trees: Array[WorkingNode]) extends Serializable {
  def predict(treePoint: TreePoint): Double = {
    if (trees.nonEmpty){
      bias + trees.map{ root =>  root.predict(treePoint) }.sum
    } else {
      bias
    }
  }

  def toSparkXGBoostModel(splitsBundle: Array[Array[Split]], loss: Loss): SparkXGBoostModel = {
    val immutableTrees = trees.map(workingNode => workingNode.toNode(splitsBundle)).toList
    new SparkXGBoostModel(bias, immutableTrees, loss)
  }
}
