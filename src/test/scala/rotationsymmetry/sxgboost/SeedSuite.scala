package rotationsymmetry.sxgboost

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.scalatest.FunSuite
import rotationsymmetry.sxgboost.loss.SquareLoss
import rotationsymmetry.sxgboost.utils.MLlibTestSparkContext
import rotationsymmetry.sxgboost.utils.TestingUtils._

class SeedSuite extends FunSuite with MLlibTestSparkContext with TestData {
  test("Different runs with the seed returns the same result"){
    val data = sqlContext.createDataFrame(randomLabelPointRDD(sc, 100, 10, 3, 999))

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(2)
      .fit(data)

    val sXGBoost1 = new SparkXGBoostRegressor(new SquareLoss)
      .setFeaturesCol("indexedFeatures")
      .setMaxDepth(5)
      .setNumTrees(1)
      .setSampleRatio(0.5)
      .setFeatureSampleRatio(0.5)
      .setSeed(999)

    val sXGBoostModel1 = sXGBoost1.fit(featureIndexer.transform(data))

    val sXGBoost2 = new SparkXGBoostRegressor(new SquareLoss)
      .setFeaturesCol("indexedFeatures")
      .setMaxDepth(5)
      .setNumTrees(1)
      .setSampleRatio(0.5)
      .setFeatureSampleRatio(0.5)
      .setSeed(999)

    val sXGBoostModel2 = sXGBoost2.fit(featureIndexer.transform(data))

    val sXGBoost3 = new SparkXGBoostRegressor(new SquareLoss)
      .setFeaturesCol("indexedFeatures")
      .setMaxDepth(5)
      .setNumTrees(1)
      .setSampleRatio(0.5)
      .setFeatureSampleRatio(0.5)
      .setSeed(998)

    val sXGBoostModel3 = sXGBoost3.fit(featureIndexer.transform(data))

    val evaluator = new RegressionEvaluator()
    val rmse1 = evaluator.evaluate(sXGBoostModel1.transform(featureIndexer.transform(data)))
    val rmse2 = evaluator.evaluate(sXGBoostModel2.transform(featureIndexer.transform(data)))
    val rmse3 = evaluator.evaluate(sXGBoostModel3.transform(featureIndexer.transform(data)))

    assert(rmse1 === rmse2)
    assert(rmse1 !~= rmse3 relTol 1e-2)
  }
}
