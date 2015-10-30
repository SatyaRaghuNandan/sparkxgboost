package rotationsymmetry.sxgboost

import org.apache.spark.ml.attribute._
import org.apache.spark.sql.types.StructField

import scala.collection.immutable.HashMap


private[sxgboost] object MetadataUtils {
  /**
   * Examine a schema to identify categorical (Binary and Nominal) features.
   *
   * @param featuresSchema  Schema of the features column.
   *                        If a feature does not have metadata, it is assumed to be continuous.
   *                        If a feature is Nominal, then it must have the number of values
   *                        specified.
   * @return  Map: feature index --> number of categories.
   *          The map's set of keys will be the set of categorical feature indices.
   */
  def getCategoricalFeatures(featuresSchema: StructField): Map[Int, Int] = {
    val metadata = AttributeGroup.fromStructField(featuresSchema)
    if (metadata.attributes.isEmpty) {
      HashMap.empty[Int, Int]
    } else {
      metadata.attributes.get.zipWithIndex.flatMap { case (attr, idx) =>
        if (attr == null) {
          Iterator()
        } else {
          attr match {
            case _: NumericAttribute | UnresolvedAttribute => Iterator()
            case binAttr: BinaryAttribute => Iterator(idx -> 2)
            case nomAttr: NominalAttribute =>
              nomAttr.getNumValues match {
                case Some(numValues: Int) => Iterator(idx -> numValues)
                case None => throw new IllegalArgumentException(s"Feature $idx is marked as" +
                  " Nominal (categorical), but it does not have the number of values specified.")
              }
          }
        }
      }.toMap
    }
  }
}
