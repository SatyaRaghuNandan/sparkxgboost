package rotationsymmetry.sxgboost

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.util.Random

/**
 * Interface for a "Split," which specifies a test made at a decision tree node
 * to choose the left or right path.
 */
sealed trait Split extends Serializable {

  /** Index of feature which this split tests */
  def featureIndex: Int

  /**
   * Return true (split to left) or false (split to right).
   * @param features  Vector of features (original values, not binned).
   */
  def shouldGoLeft(features: Vector): Boolean

}

class OrderedSplit(override val featureIndex: Int, val threshold: Double) extends Split {

  override def shouldGoLeft(features: Vector): Boolean = {
    features(featureIndex) <= threshold
  }
}

object OrderedSplit {
  def createOrderedSplits(
       input: RDD[LabeledPoint],
       categoricalFeatures: Map[Int, Int],
       maxBins: Int,
       seed: Long = new Random().nextLong()): Array[Array[Split]] = {

    val requiredSamples = math.max(maxBins * maxBins, 10000)
    val numSamples = input.count()
    val fraction = if (requiredSamples < numSamples) {
      requiredSamples.toDouble / numSamples
    } else {
      1.0
    }

    val sampledInput = input.sample(false, fraction, seed).collect()

    val numFeatures = sampledInput.head.features.size
    val splits = new Array[Array[Split]](numFeatures)

    // Find all splits.
    // Iterate over all features.
    var featureIndex = 0
    while (featureIndex < numFeatures) {
      if (!categoricalFeatures.isDefinedAt(featureIndex)) {
        val featureSamples = sampledInput.map(_.features(featureIndex))
        val featureSplits = findSplitsForContinuousFeature(featureSamples, maxBins)

        val numSplits = featureSplits.length
        splits(featureIndex) = new Array[Split](numSplits)

        var splitIndex = 0
        while (splitIndex < numSplits) {
          val threshold = featureSplits(splitIndex)
          splits(featureIndex)(splitIndex) = new OrderedSplit(featureIndex, threshold)
          splitIndex += 1
        }
      } else {
        val numSplits = categoricalFeatures(featureIndex)
        splits(featureIndex) = new Array[Split](numSplits)

        var splitIndex = 0
        while (splitIndex < numSplits) {
          splits(featureIndex)(splitIndex) = new OrderedSplit(featureIndex, splitIndex)
          splitIndex += 1
        }
      }
      featureIndex += 1
    }
    splits
  }


  def findSplitsForContinuousFeature(featureSamples: Array[Double], maxBins: Int): Array[Double] = {

    val numSplits = maxBins

    // get count for each distinct value
    val valueCountMap = featureSamples.foldLeft(Map.empty[Double, Int]) { (m, x) =>
      m + ((x, m.getOrElse(x, 0) + 1))
    }
    // sort distinct values
    val valueCounts = valueCountMap.toSeq.sortBy(_._1).toArray

    // if possible splits is not enough or just enough, just return all possible splits
    val possibleSplits = valueCounts.length
    if (possibleSplits <= numSplits) {
      valueCounts.map(_._1)
    } else {
      // stride between splits
      val stride: Double = featureSamples.length.toDouble / (numSplits + 1)

      // iterate `valueCount` to find splits
      val splitsBuilder = mutable.ArrayBuilder.make[Double]
      var index = 1
      // currentCount: sum of counts of values that have been visited
      var currentCount = valueCounts(0)._2
      // targetCount: target value for `currentCount`.
      // If `currentCount` is closest value to `targetCount`,
      // then current value is a split threshold.
      // After finding a split threshold, `targetCount` is added by stride.
      var targetCount = stride
      while (index < valueCounts.length) {
        val previousCount = currentCount
        currentCount += valueCounts(index)._2
        val previousGap = math.abs(previousCount - targetCount)
        val currentGap = math.abs(currentCount - targetCount)
        // If adding count of current value to currentCount
        // makes the gap between currentCount and targetCount smaller,
        // previous value is a split threshold.
        if (previousGap < currentGap) {
          splitsBuilder += valueCounts(index - 1)._1
          targetCount += stride
        }
        index += 1
      }

      splitsBuilder.result()
    }

  }
}


