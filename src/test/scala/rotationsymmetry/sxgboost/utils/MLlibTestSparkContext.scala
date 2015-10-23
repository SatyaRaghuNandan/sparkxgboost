package rotationsymmetry.sxgboost.utils

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfterAll, Suite}

trait MLlibTestSparkContext extends BeforeAndAfterAll { self: Suite =>
  @transient var sc: SparkContext = _
  @transient var sqlContext: SQLContext = _

  override def beforeAll() {
    super.beforeAll()
    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("MLlibUnitTest")
    sc = new SparkContext(conf)
    sqlContext = new SQLContext(sc)
  }

  override def afterAll() {
    sqlContext = null
    if (sc != null) {
      sc.stop()
    }
    sc = null
    super.afterAll()
  }
}
