package rotationsymmetry.sxgboost

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.scalatest.FunSuite

class TreePointSuite extends FunSuite{

  test("findBin") {
    val labeledPoint = LabeledPoint(0, Vectors.dense(0.0, 1.0, 1.5, 2.1))
    val thresholds = Array[Double](1.0, 2.0)
    assert(TreePoint.findBin(0, labeledPoint, thresholds) == 0)
    assert(TreePoint.findBin(1, labeledPoint, thresholds) == 0)
    assert(TreePoint.findBin(2, labeledPoint, thresholds) == 1)
    assert(TreePoint.findBin(3, labeledPoint, thresholds) == 2)
  }

}
