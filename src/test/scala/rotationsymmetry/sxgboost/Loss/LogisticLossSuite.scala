package rotationsymmetry.sxgboost.loss

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.scalatest.FunSuite
import rotationsymmetry.sxgboost.utils.MLlibTestSparkContext
import rotationsymmetry.sxgboost.utils.TestingUtils._

class LogisticLossSuite extends FunSuite with MLlibTestSparkContext with NumericDiff{
  val loss = new LogisticLoss()

  def numericLoss(label: Double, f: Double): Double = {
    - label * f + Math.log(1 + Math.exp(f))
  }

  test("diff's match numerical counterparts") {
    val delta =0.001
    Seq[Double](0.0, 1.0).foreach { label =>
      Seq[Double](-1.5, -0.8, 0, 0.8, 1.5).foreach { f =>
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

    val p = data.map(lp => lp.label).sum / data.length
    val theta = Math.log(p / (1 - p))

    val rdd = sc.parallelize(data, 2)
    assert(loss.getInitialBias(rdd) ~== theta relTol 1e-5)
  }

  test("prediction from score"){
    assert(loss.toPrediction(0.0) ~== 0.5 relTol 1e-5)
    assert(loss.toPrediction(1.0) ~== 0.7310586 relTol 1e-5)
    assert(loss.toPrediction(-1.0) ~== 0.2689414 relTol 1e-5)
  }
}
