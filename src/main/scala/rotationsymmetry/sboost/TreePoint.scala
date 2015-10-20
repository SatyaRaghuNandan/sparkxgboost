package rotationsymmetry.sboost

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

case class TreePoint(label: Double, binnedFeature: Array[Int])

object TreePoint {
  def convertFromLabeledPoints(input: RDD[LabeledPoint], splits: Array[Array[Split]]): RDD[TreePoint] = {
    null
  }
}
