package rotationsymmetry.sxgboost

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkContext, SparkConf}
import org.scalatest.FunSuite


class SparkXGBoostSuite extends FunSuite{
  test("SXG") {
    val conf = new SparkConf().setAppName("Test").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    val data = MLUtils.loadLibSVMFile(sc, "sample_libsvm_data.txt").toDF()

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(2)
      .fit(data)

    val sXGBoost = new SparkXGBoost()
      .setFeaturesCol("indexedFeatures")
      .setLoss(new SquareLoss)
      .setMaxDepth(0)
      .setNumTrees(1)


    val model = sXGBoost.train(featureIndexer.transform(data))

    val predictions = model.predict(featureIndexer.transform(data))

    val evaluator = new RegressionEvaluator()
    val mse = evaluator.evaluate(predictions)
    print(mse)
  }
}