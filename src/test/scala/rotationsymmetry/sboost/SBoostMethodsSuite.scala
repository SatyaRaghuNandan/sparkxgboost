package rotationsymmetry.sboost

import org.scalatest.FunSuite

class SBoostMethodsSuite extends FunSuite{

  val sboost = new SBoost()

  test("extractDiffsFromStatsView") {
    val v = Seq[Double](1, 2, 3, 4, 5, 6)
    val (d1, d2) = sboost.extractDiffsFromStatsView(v)
    assert(d1 === Array[Double](1, 3, 5))
    assert(d2 === Array[Double](2, 4, 6))
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

}
