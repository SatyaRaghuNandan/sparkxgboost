package rotationsymmetry.sxgboost.Loss

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.scalatest.FunSuite
import rotationsymmetry.sxgboost.loss.PoissonLoss
import rotationsymmetry.sxgboost.utils.MLlibTestSparkContext
import rotationsymmetry.sxgboost.utils.TestingUtils._

class PoissonLossSuite extends FunSuite with MLlibTestSparkContext with NumericDiff {
  val loss = new PoissonLoss()

  /**
   * The term -log(factorial(label)) is ignore since it
   * does not contribute to the diff's.
  */
  def numericLoss(label: Double, f: Double): Double = {
    label * Math.log(f) -f
  }

  test("diff's match numerical counterparts") {
    val delta = 0.0001
    Seq[Double](0.0, 1.0, 2.0, 3.0).foreach { label =>
      Seq[Double](0.1, 0.8, 1.5).foreach { f =>
        assert(numericDiff1(label, f, delta) ~== loss.diff1(label, f) relTol 1e-3)
        assert(numericDiff2(label, f, delta) ~== loss.diff2(label, f) relTol 1e-3)
      }
    }
  }

  test("initial bias") {
    val data = Seq(
      LabeledPoint(1.0, Vectors.dense(0.0)),
      LabeledPoint(1.0, Vectors.dense(0.0)),
      LabeledPoint(0.0, Vectors.dense(0.0))
    )

    val mean = data.map(lp => lp.label).sum / data.length

    val rdd = sc.parallelize(data, 2)
    assert(loss.getInitialBias(rdd) ~== mean relTol 1e-5)
  }

  test("prediction from score") {
    assert(loss.toPrediction(0.0) === 0.0)
    assert(loss.toPrediction(1.0) === 1.0)
    assert(loss.toPrediction(1.5) === 1.5)
  }
}