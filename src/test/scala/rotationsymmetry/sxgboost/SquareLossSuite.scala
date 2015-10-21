package rotationsymmetry.sxgboost

import org.scalatest.FunSuite
import TestingUtils._

class SquareLossSuite extends FunSuite{

  val squareLoss = new SquareLoss

  test("approximation should match truth for square loss") {
    val label = 3.0
    val f = 2.5
    val delta = 0.1
    val diff0 = label - f
    val diff = label - (f + delta)
    val trueLoss0 = diff0 * diff0
    val trueLoss = diff * diff
    val approx = trueLoss0 + squareLoss.diff1(label, f) * delta + 0.5 * squareLoss.diff2(label, f) * delta * delta
    assert(trueLoss ~== approx relTol 1e-5)
  }

}
