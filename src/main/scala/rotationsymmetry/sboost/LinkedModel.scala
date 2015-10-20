package rotationsymmetry.sboost


class Loss {
  def diff1(label: Double, f: Double): Double = 0.0

  def diff2(label: Double, f: Double): Double = 0.0

}


class SquareLoss extends Loss{
  override def diff1(label: Double, f: Double): Double = 2 * (f - label)

  override def diff2(label: Double, f: Double): Double = 2.0
}