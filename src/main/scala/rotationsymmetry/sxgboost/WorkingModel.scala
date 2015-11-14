package rotationsymmetry.sxgboost

private[sxgboost] class WorkingModel(val bias: Double, var trees: Array[WorkingNode]) extends Serializable {
  def predict(treePoint: TreePoint): Double = {
    if (trees.nonEmpty){
      bias + trees.map{ root =>  root.predict(treePoint) }.sum
    } else {
      bias
    }
  }

  def getImmutableTrees(splitsBundle: Array[Array[Split]]): List[Node] = {
    trees.map(workingNode => workingNode.toNode(splitsBundle)).toList
  }

  def toTrainedModel(splitsBundle: Array[Array[Split]]): TrainedModel = {
    val immutableTrees = getImmutableTrees(splitsBundle)
    TrainedModel(bias, immutableTrees)
  }
}
