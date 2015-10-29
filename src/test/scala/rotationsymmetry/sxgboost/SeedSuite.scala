package rotationsymmetry.sxgboost

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.scalatest.FunSuite
import rotationsymmetry.sxgboost.loss.SquareLoss
import rotationsymmetry.sxgboost.utils.MLlibTestSparkContext

class SeedSuite extends FunSuite with MLlibTestSparkContext with TestData {
  test("Different runs with the seed returns the same result"){
    val data = sqlContext.createDataFrame(randomLabelPointRDD(sc, 100, 10, 3, 999))

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(2)
      .fit(data)

    val sXGBoost1 = new SparkXGBoost(new SquareLoss)
      .setFeaturesCol("indexedFeatures")
      .setMaxDepth(5)
      .setNumTrees(1)
      .setSeed(1)

    val sXGBoostModel1 = sXGBoost1.fit(featureIndexer.transform(data))

    val sXGBoost2 = new SparkXGBoost(new SquareLoss)
      .setFeaturesCol("indexedFeatures")
      .setMaxDepth(5)
      .setNumTrees(1)
      .setSeed(1)

    val sXGBoostModel2 = sXGBoost2.fit(featureIndexer.transform(data))

    val evaluator = new RegressionEvaluator()
    val rmse1 = evaluator.evaluate(sXGBoostModel1.transform(featureIndexer.transform(data)))
    val rmse2 = evaluator.evaluate(sXGBoostModel2.transform(featureIndexer.transform(data)))

    assert(rmse1 === rmse2)
  }
}
