package rotationsymmetry.sxgboost.loss

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

class SquareLoss extends Loss {
  override def diff1(label: Double, f: Double): Double = 2 * (f - label)

  override def diff2(label: Double, f: Double): Double = 2.0

  override def toPrediction(score: Double): Double = score

  override def getInitialBias(input: RDD[LabeledPoint]): Double = {
    val totalWeight = input.count()
    val scaledLabels = input.map(lp => lp.label / totalWeight)
    scaledLabels.treeReduce(_+_)
  }
}
