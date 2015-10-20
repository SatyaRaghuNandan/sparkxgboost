package rotationsymmetry.sboost

class WorkingNode {
  var leftChild: Option[WorkingNode] = None
  var rightChild: Option[WorkingNode] = None
  var split: Option[WorkingSplit] = None
  var prediction: Option[Double] = None
  var idxInBatch: Option[Int] = None

  def isLeaf: Boolean = split.isEmpty

  def predict(treePoint: TreePoint): Double = {
    if (isLeaf) {
      prediction.get
    } else {
      if (split.get.shouldGoLeft(treePoint.binnedFeature)) {
        leftChild.get.predict(treePoint)
      } else {
        rightChild.get.predict(treePoint)
      }
    }
  }

  def locateNode(treePoint: TreePoint): WorkingNode = {
    if (isLeaf) {
      this
    } else {
      if (split.get.shouldGoLeft(treePoint.binnedFeature)) {
        leftChild.get.locateNode(treePoint)
      } else {
        rightChild.get.locateNode(treePoint)
      }
    }
  }
}
