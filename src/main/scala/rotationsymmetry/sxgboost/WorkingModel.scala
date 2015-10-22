package rotationsymmetry.sxgboost

case class WorkingModel(trees: Array[WorkingNode]) extends Serializable {
  def predict(treePoint: TreePoint): Double = {
    trees.map{ root =>  root.predict(treePoint) }.sum
  }

  def toSparkXGBoostModel(splitsBundle: Array[Array[Split]], loss: Loss): SparkXGBoostModel = {
    val immutableTrees = trees.map(workingNode => workingNode.toNode(splitsBundle)).toList
    new SparkXGBoostModel(immutableTrees, loss)
  }
}
