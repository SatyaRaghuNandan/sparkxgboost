package rotationsymmetry.sxgboost.loss

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
 * Loss for logistic classification
 * Assume we observe poisson count label =  0, 1, 2, ...  and the score suggested by
 * the model is f, then the loss (e.g. negative log likelihood)
 * is log(label!) - label * log(f) + f.
 */
class PoissonLoss extends Loss{
  override def diff1(label: Double, f: Double): Double = - label / f + 1

  override def diff2(label: Double, f: Double): Double = label / Math.pow(f, 2.0)

  override def toPrediction(score: Double): Double = score

  override def getInitialBias(input: RDD[LabeledPoint]): Double = {
    val totalWeight = input.count()
    val scaledLabels = input.map(lp => lp.label / totalWeight)
    scaledLabels.treeReduce(_+_)
  }
}
