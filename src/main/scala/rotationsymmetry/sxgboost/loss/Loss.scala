package rotationsymmetry.sxgboost.loss

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD


abstract class Loss extends Serializable{
  def diff1(label: Double, f: Double): Double

  def diff2(label: Double, f: Double): Double

  /*
  Classification: prediction is probability
  Regression: prediction is label
   */
  def toPrediction(score: Double): Double

  def getInitialBias(input: RDD[LabeledPoint]): Double

}


