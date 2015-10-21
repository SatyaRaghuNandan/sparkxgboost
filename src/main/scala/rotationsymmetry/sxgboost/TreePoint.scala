package rotationsymmetry.sxgboost

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

case class TreePoint(label: Double, binnedFeature: Array[Int])

object TreePoint {
  def convertToTreeRDD(
                        input: RDD[LabeledPoint],
                        splitsBundle: Array[Array[Split]]): RDD[TreePoint] = {

    val thresholds: Array[Array[Double]] = splitsBundle.map { splits =>
      splits.map(_.asInstanceOf[OrderedSplit].threshold)
    }
    input.map { x =>
      TreePoint.labeledPointToTreePoint(x, thresholds)
    }
  }



  def labeledPointToTreePoint(
                               labeledPoint: LabeledPoint,
                               thresholds: Array[Array[Double]]): TreePoint = {
    val numFeatures = labeledPoint.features.size
    val bins = new Array[Int](numFeatures)
    var featureIndex = 0
    while (featureIndex < numFeatures) {
      bins(featureIndex) =
        findBin(featureIndex, labeledPoint, thresholds(featureIndex))
      featureIndex += 1
    }
    new TreePoint(labeledPoint.label, bins)
  }

  def findBin(
               featureIndex: Int,
               labeledPoint: LabeledPoint,
               thresholds: Array[Double]): Int = {
    val featureValue = labeledPoint.features(featureIndex)

    val idx = java.util.Arrays.binarySearch(thresholds, featureValue)
    if (idx >= 0) {
      idx
    } else {
      -idx - 1
    }
  }
}
