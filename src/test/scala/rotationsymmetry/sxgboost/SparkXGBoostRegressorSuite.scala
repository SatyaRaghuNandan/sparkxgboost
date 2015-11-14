package rotationsymmetry.sxgboost

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.scalatest.FunSuite
import rotationsymmetry.sxgboost.loss.SquareLoss
import rotationsymmetry.sxgboost.utils.MLlibTestSparkContext
import rotationsymmetry.sxgboost.utils.TestingUtils._


class SparkXGBoostRegressorSuite extends FunSuite with TestData with MLlibTestSparkContext {
  test("Compare with DecisionTree using simple data") {

    val data = sqlContext.createDataFrame(sc.parallelize(simpleData, 2))

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(2)
      .fit(data)

    val sparkXGBoostRegressor = new SparkXGBoostRegressor(new SquareLoss)
      .setFeaturesCol("indexedFeatures")
      .setMaxDepth(1)
      .setNumTrees(1)
    val sparkXGBoostPipeline = new Pipeline()
      .setStages(Array(featureIndexer, sparkXGBoostRegressor))
    val sXGBoostModel = sparkXGBoostPipeline.fit(data)

    val dt = new DecisionTreeRegressor()
      .setFeaturesCol("indexedFeatures")
      .setMaxDepth(1)
    val dtPipeLine = new Pipeline()
      .setStages(Array(featureIndexer, dt))
    val dtModel = dtPipeLine.fit(data)

    val evaluator = new RegressionEvaluator()
    val sXGBoostrmse = evaluator.evaluate(sXGBoostModel.transform(data))
    val dtrmse = evaluator.evaluate(dtModel.transform(data))

    assert(sXGBoostrmse ~== dtrmse relTol 1e-5)
  }

  test("Compare with DecisionTree using random data") {

    val data = sqlContext.createDataFrame(randomLabelPointRDD(sc, 40, 10, 2, 999))

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(2)
      .fit(data)

    val sparkXGBoostRegressor = new SparkXGBoostRegressor(new SquareLoss)
      .setFeaturesCol("indexedFeatures")
      .setMaxDepth(5)
      .setNumTrees(1)
    val sparkXGBoostPipeline = new Pipeline()
      .setStages(Array(featureIndexer, sparkXGBoostRegressor))
    val sXGBoostModel = sparkXGBoostPipeline.fit(data)

    val dt = new DecisionTreeRegressor()
      .setFeaturesCol("indexedFeatures")
      .setMaxDepth(5)
    val dtPipeLine = new Pipeline()
      .setStages(Array(featureIndexer, dt))
    val dtModel = dtPipeLine.fit(data)

    val evaluator = new RegressionEvaluator()
    val sXGBoostrmse = evaluator.evaluate(sXGBoostModel.transform(data))
    val dtrmse = evaluator.evaluate(dtModel.transform(data))

    assert(sXGBoostrmse ~== dtrmse relTol 1e-5)
  }
}
