package rotationsymmetry.sxgboost

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.{SparkContext, SparkConf}
import org.scalatest.FunSuite
import rotationsymmetry.sxgboost.loss.SquareLoss
import rotationsymmetry.sxgboost.utils.{TestingUtils, MLlibTestSparkContext}
import TestingUtils._
import rotationsymmetry.sxgboost.utils.MLlibTestSparkContext

class WithDecisionTreeSuite extends FunSuite with TestData with MLlibTestSparkContext{

  test("Compare with DecisionTree using simple data") {

    val data = sqlContext.createDataFrame(sc.parallelize(simpleData, 2))

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(2)
      .fit(data)

    val sXGBoost = new SparkXGBoost(new SquareLoss)
      .setFeaturesCol("indexedFeatures")
      .setMaxDepth(1)
      .setNumTrees(1)
    val sXGBoostModel = sXGBoost.fit(featureIndexer.transform(data))

    val dt = new DecisionTreeRegressor()
      .setFeaturesCol("indexedFeatures")
      .setMaxDepth(1)
    val dtModel = dt.fit(featureIndexer.transform(data))

    val evaluator = new RegressionEvaluator()
    val sXGBoostrmse = evaluator.evaluate(sXGBoostModel.transform(featureIndexer.transform(data)))
    val dtrmse = evaluator.evaluate(dtModel.transform(featureIndexer.transform(data)))

    assert(sXGBoostrmse ~== dtrmse relTol 1e-5)
  }

  test("Compare with DecisionTree using random data") {

    val data = sqlContext.createDataFrame(randomLabelPointRDD(sc, 20, 10, 2, 999))

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(2)
      .fit(data)

    val sXGBoost = new SparkXGBoost(new SquareLoss)
      .setFeaturesCol("indexedFeatures")
      .setMaxDepth(5)
      .setNumTrees(1)
    val sXGBoostModel = sXGBoost.fit(featureIndexer.transform(data))

    val dt = new DecisionTreeRegressor()
      .setFeaturesCol("indexedFeatures")
      .setMaxDepth(5)
    val dtModel = dt.fit(featureIndexer.transform(data))

    val evaluator = new RegressionEvaluator()
    val sXGBoostrmse = evaluator.evaluate(sXGBoostModel.transform(featureIndexer.transform(data)))
    val dtrmse = evaluator.evaluate(dtModel.transform(featureIndexer.transform(data)))

    assert(sXGBoostrmse ~== dtrmse relTol 1e-5)
  }

}
