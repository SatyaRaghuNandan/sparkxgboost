package rotationsymmetry.sxgboost

import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkContext, SparkConf}
import org.scalatest.FunSuite


class SparkXGBoostSuite extends FunSuite{
  test("GBT") {
    val conf = new SparkConf().setAppName("Test").setMaster("local[2]")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    val data = MLUtils.loadLibSVMFile(sc, "sample_libsvm_data.txt").toDF()

    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)

    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    val sXGBoost = new SparkXGBoost()
      .setFeaturesCol("indexedFeatures")
      .setLoss(new SquareLoss)
      .setNumTrees(10)

    val indexedTrainingData = featureIndexer.transform(trainingData)
    val indexedTestData = featureIndexer.transform(testData)

    val model = sXGBoost.train(indexedTrainingData)

    val predictions = model.predict(indexedTestData)
    val x = 1
  }
}