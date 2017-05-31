package math

import actors.Activation

import scala.language.postfixOps
import scala.util.Random

case class Vector(xs: Double*) {

  def size: Int = xs.size

  def sum: Double = xs.sum

  /*
  dot product
   */
  def dot(other: Vector): Double = {
    require(other.size == this.size)
    xs zip other.xs map { case (x1, x2) => x1 * x2 } sum
  }

  def -(other: Vector): Vector = {
    require(other.size == this.size)
    Vector(xs zip other.xs map { case (x1, x2) => x1 - x2 } toList)
  }

  def -(d: Double): Vector =
    Vector(xs map { x =>
      x - d
    } toList)

  /*
  append
   */
  def :+(x: Double): Vector = Vector(xs :+ x toList)

  /*
  update
   */
  def updated(i: Int, v: Double): Vector =
    Vector(xs.updated(i, v) toList)

  def output(): Seq[Activation] =
    xs.zipWithIndex.map { case (x, i) => Activation(i, x) }

}

object Vector {
  def randn(size: Int): Vector = {
    val l = for (_ <- 1 to size) yield {
      Random.nextGaussian()
    }
    Vector(l toList)
  }

  def zeroes(size: Int): Vector = Vector(List.fill(size)(0.0))

  def oneHotEncoded(x: Int, base: Int = 10): Vector =
    zeroes(base).updated(x, 1.0)

  def apply(m: Map[Int, Double]): Vector = {
    val keys = m.keys.toList.sorted
    require(keys == (0 until m.size).toList)
    Vector(keys.map(k => m(k)))
  }

  def apply(xs: List[Double]): Vector = Vector(xs: _*)
}
