package rotationsymmetry.sxgboost

case class WorkingSplit(featureIndex: Int, threshold: Int) extends Serializable {
  def shouldGoLeft(binnedFeatures: Array[Int]): Boolean = {
    binnedFeatures(featureIndex) <= threshold
  }
}

case class SplitInfo(
     split: WorkingSplit,
     gain: Double,
     leftPrediction: Double,
     rightPrediction: Double,
     leftWeight: Double,
     rightWeight: Double)