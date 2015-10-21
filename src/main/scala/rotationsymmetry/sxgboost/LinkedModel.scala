package rotationsymmetry.sxgboost


abstract class Loss {
  def diff1(label: Double, f: Double): Double

  def diff2(label: Double, f: Double): Double

  def toPrediction(score: Double): Double
}


class SquareLoss extends Loss{
  override def diff1(label: Double, f: Double): Double = 2 * (f - label)

  override def diff2(label: Double, f: Double): Double = 2.0

  override def toPrediction(score: Double): Double = score
}