package rotationsymmetry.sxgboost

private[sxgboost] case class WorkingSplit(featureIndex: Int, threshold: Int) extends Serializable {
  def shouldGoLeft(binnedFeatures: Array[Int]): Boolean = {
    binnedFeatures(featureIndex) <= threshold
  }
}

private[sxgboost] case class SplitInfo(
    split: WorkingSplit,
    gain: Double,
    leftPrediction: Double,
    rightPrediction: Double,
    leftWeight: Double,
    rightWeight: Double)