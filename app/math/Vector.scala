package math

import scala.language.postfixOps
import scala.util.Random

case class Vector(xs: Double*) {

  def size: Int = xs.size

  def sum: Double = xs.sum

  /*
  dot product
   */
  def *(other: Vector): Double = {
    require(other.size == this.size)
    xs zip other.xs map { case (x1, x2) => x1 * x2 } sum
  }

  def sigmoid(other: Vector, bias: Double): Double =
    1 / (1 + Math.exp(-(this * other + bias)))

  /*
  append
   */
  def :+(x: Double): Vector = Vector(xs :+ x: _*)

  /*
  update
   */
  def updated(i: Int, v: Double): Vector =
    Vector(xs.updated(i, v): _*)

}

object Vector {
  def randn(size: Int): Vector = {
    val l = for (i <- 1 to size) yield {
      Random.nextGaussian()
    }
    Vector(l: _*)
  }

  def apply(m: Map[Int, Double]): Vector = {
    val keys = m.keys.toList.sorted
    require(keys == (0 until m.size).toList)
    Vector(keys.map(k => m(k)): _*)
  }
}
