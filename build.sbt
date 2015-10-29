name := "SparkXGBoost"

spName := "rotationsymmetry/SparkXGBoost"

organization := "rotationsymmetry"

version := "0.2.0"

scalaVersion := "2.10.6"

sparkVersion := sys.props.getOrElse("spark.testVersion", "1.5.1") 

sparkComponents ++= Seq("core", "mllib")

spAppendScalaVersion := true

parallelExecution := false

libraryDependencies ++= Seq(
  "org.scalatest" % "scalatest_2.10" % "2.2.4" % "test"
)

credentials += Credentials(Path.userHome / "Documents" / "git" / "token")

licenses += "Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0")
