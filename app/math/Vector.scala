package math

import scala.language.postfixOps

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
    1 / (1 + Math.exp(- (this * other + bias) ))

  /*
  append
   */
  def :+(x: Double): Vector = Vector(xs :+ x: _*)

}
