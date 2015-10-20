package rotationsymmetry.sboost

import org.scalatest.{BeforeAndAfter, FunSuite}
import TestingUtils._

class SBoostMethodsSuite extends FunSuite with BeforeAndAfter{

  var sboost: SBoost = null

  before {
    sboost = new SBoost()
  }

  test("extractDiffsFromStatsView") {
    val v = Seq[Double](1, 2, 10, 3, 4, 20, 5, 6, 30)
    val (d1, d2, weights) = sboost.extractDiffsAndWeightsFromStatsView(v)
    assert(d1 === Array[Double](1, 3, 5))
    assert(d2 === Array[Double](2, 4, 6))
    assert(weights === Array[Double](10, 20, 30))
  }

  test("getCuSumAndTotal") {
    val ds =Seq[Double](1, 3, 2)
    val (cusum, total) = sboost.getCuSumAndTotal(ds)
    assert(cusum === Array(1, 4))
    assert(total === 6)
  }

  test("sampleFeatureIndices") {
    val numFeatures = 10
    val numSampledFeatures = 5
    val numSamples = 20
    val featureIndicesBundle = sboost.sampleFeatureIndices(numFeatures, numSampledFeatures, numSamples)
    assert(featureIndicesBundle.length == numSamples)
    featureIndicesBundle.foreach(indices => assert(indices.length == numSampledFeatures))
  }

  test("findBestSplitForSingleFeature") {
    val statsView = Seq(1.0, 2.0, 0.0, 0.0, 3.0, 6.0, 1.5, 9.0)
    val lambda = 0.5
    val optimSplit = sboost.findBestSplitForSingleFeature(statsView, 1, lambda)
    //assert(optimSplit.split.threshold == 2)
    //assert(optimSplit.gain ~== 1.7 relTol 1e-1)
  }

  test("findBestSplit"){

  }
}
