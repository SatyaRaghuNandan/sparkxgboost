package rotationsymmetry.sxgboost.loss

trait NumericDiff {
  def numericLoss(label: Double, f: Double): Double

  def numericDiff1(label: Double, f: Double, delta: Double): Double = {
    (numericLoss(label, f + delta) - numericLoss(label, f)) / delta
  }

  def numericDiff2(label: Double, f: Double, delta: Double): Double = {
    (numericLoss(label, f + delta) - 2 * numericLoss(label, f) + numericLoss(label, f - delta)) /
      Math.pow(delta, 2)
  }
}
