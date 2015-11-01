package rotationsymmetry.sxgboost.loss

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
 * Loss for logistic classification
 * Assume we observe group label y = 0 or 1 and the score suggested by
 * the model is f, then the loss (e.g. negative log likelihood) is y*f+log(1+exp(f))
 */
class LogisticLoss extends Loss {
  override def diff1(label: Double, f: Double): Double = {
    val e = Math.exp(f)
    - label + e / (1 + e)
  }

  override def diff2(label: Double, f: Double): Double = {
    val e = Math.exp(f)
    e / Math.pow(1 + e, 2)
  }

  override def toPrediction(score: Double): Double = {
    if (score >= 0.5) {
      1.0
    } else {
      0
    }
  }

  override def getInitialBias(input: RDD[LabeledPoint]): Double = {
    val totalWeight = input.count()
    val scaledLabels = input.map(lp => lp.label / totalWeight)
    val p = scaledLabels.treeReduce(_+_)
    Math.log(p / (1 - p))
  }

}
