package rotationsymmetry.sxgboost

import org.scalatest.{BeforeAndAfter, FunSuite}
import TestingUtils._

class SparkXGBoostMethodsSuite extends FunSuite with BeforeAndAfter{

  var sparkXGBoost: SparkXGBoost = null

  before {
    sparkXGBoost = new SparkXGBoost()
  }

  test("extractDiffsFromStatsView") {
    val v = Seq[Double](1, 2, 10, 3, 4, 20, 5, 6, 30)
    val (d1, d2, weights) = sparkXGBoost.extractDiffsAndWeightsFromStatsView(v)
    assert(d1 === Array[Double](1, 3, 5))
    assert(d2 === Array[Double](2, 4, 6))
    assert(weights === Array[Double](10, 20, 30))
  }

  test("getCuSumAndTotal") {
    val ds =Seq[Double](1, 3, 2)
    val (cusum, total) = sparkXGBoost.getCuSumAndTotal(ds)
    assert(cusum === Array(1, 4))
    assert(total === 6)
  }

  test("sampleFeatureIndices") {
    val numFeatures = 10
    val numSampledFeatures = 5
    val numSamples = 20
    val featureIndicesBundle = sparkXGBoost.sampleFeatureIndices(numFeatures, numSampledFeatures, numSamples)
    assert(featureIndicesBundle.length == numSamples)
    featureIndicesBundle.foreach(indices => assert(indices.length == numSampledFeatures))
  }

  test("findBestSplitForSingleFeature") {
    /*
    See Scenario 1 of
    https://docs.google.com/spreadsheets/d/1rPOorXThwn3XLnCRFImExSmlTSr4evBDAdY4oXMtf3A/edit?usp=sharing
    for the worksheet to derive the results.
     */
    val statsView: Seq[Double] = Seq(
      1.0, 2.0, 1.0,
      1.5, 0.2, 1.0,
      0.1, 0.0, 1.0,
      3.0, 10,  1.0,
      1.5, 9.0, 1.0
    )
    val lambda = 0.5
    val featureIndex = 1
    val optimSplit = sparkXGBoost.findBestSplitForSingleFeature(statsView, featureIndex, lambda)
    assert(optimSplit.split.threshold == 2)
    assert(optimSplit.split.featureIndex == featureIndex)
    assert(optimSplit.gain ~== 0.61 relTol 1e-1)
    assert(optimSplit.leftPrediction ~== -0.96 relTol 1e-1)
    assert(optimSplit.rightPrediction ~== -0.23 relTol 1e-1)
    assert(optimSplit.leftWeight ~== 3.0 relTol 1e-5)
    assert(optimSplit.rightWeight ~== 2.0 relTol 1e-5)
  }

  test("findBestSplit with small gamma"){
    /*
    See Scenario 1 and 2 of
    https://docs.google.com/spreadsheets/d/1rPOorXThwn3XLnCRFImExSmlTSr4evBDAdY4oXMtf3A/edit?usp=sharing
    for the worksheet to derive the results.
    */
    val stats: Array[Double] = Array(
      1.0, 2.0, 1.0,
      1.5, 0.2, 1.0,
      0.1, 0.0, 1.0,
      3.0, 10,  1.0,
      1.5, 9.0, 1.0,

      1.5, 20.0,1.0,
      3.5, 0.2, 1.0,
      1.3, 2.0, 1.0,
      0.1, 0.0, 1.0,
      3.0, 10.0,1.0,
      1.5, 9.0, 1.0
    )
    val offsets = Array[Int](0, 15)
    val featureIndices = Array[Int](0, 1)
    val lambda = 0.5
    val gamma = -10.0

    val optimSplit = sparkXGBoost.findBestSplit(stats, featureIndices, offsets, lambda, gamma)
    assert(optimSplit.get.split.featureIndex == 1)
    assert(optimSplit.get.split.threshold == 0)

  }

  test("findBestSplit with big gamma"){
    /*
    See Scenario 1 and 2 of
    https://docs.google.com/spreadsheets/d/1rPOorXThwn3XLnCRFImExSmlTSr4evBDAdY4oXMtf3A/edit?usp=sharing
    for the worksheet to derive the results.
    */
    val stats: Array[Double] = Array(
      1.0, 2.0, 1.0,
      1.5, 0.2, 1.0,
      0.1, 0.0, 1.0,
      3.0, 10,  1.0,
      1.5, 9.0, 1.0,

      1.5, 20.0,1.0,
      3.5, 0.2, 1.0,
      1.3, 2.0, 1.0,
      0.1, 0.0, 1.0,
      3.0, 10.0,1.0,
      1.5, 9.0, 1.0
    )
    val offsets = Array[Int](0, 15)
    val featureIndices = Array[Int](0, 1)
    val lambda = 0.5
    val gamma = 10.0

    val optimSplit = sparkXGBoost.findBestSplit(stats, featureIndices, offsets, lambda, gamma)
    assert(optimSplit.isEmpty)
  }
}
