package rotationsymmetry.sxgboost

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

class MetaData(
      val numFeatures: Int,
      val numBins: Array[Int]) {
}

object MetaData {

  def getMetaData(input: RDD[LabeledPoint], splits: Array[Array[Split]]): MetaData = {
    val numFeatures = input.first().features.size
    val numBins = splits.map(_.length)
    new MetaData(numFeatures, numBins)
  }
}