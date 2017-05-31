package nn

import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.sigmoid
import breeze.stats.distributions.Rand

import scala.language.postfixOps

class NeuralNet(topology: List[Int]) {

  type Vector = DenseVector[Double]
  type Matrix = DenseMatrix[Double]

  val biases: List[Vector] = topology.tail.map { size =>
    DenseVector.rand(size, Rand.gaussian)
  }

  val weights: List[Matrix] = topology
    .sliding(2)
    .map(t => DenseMatrix.rand(t(1), t.head, Rand.gaussian))
    .toList

  def backProp(x: Vector, y: Vector): (List[Vector], List[Matrix]) = {

    val nablaBiases: List[Vector] = topology.tail.map { size =>
      DenseVector.zeros[Double](size)
    }

    val nablaWeights: List[Matrix] = topology
      .sliding(2)
      .map(t => DenseMatrix.zeros[Double](t(1), t.head))
      .toList

    /*
    forward pass
     */
    val (zs, activations) = feedForward(x)

    var delta = (activations.last - y) * sigmoidPrime(zs.last)

    nablaBiases.last :=  delta
    nablaWeights.last := delta * activations(activations.size - 2).t

    for (layer <- topology.size - 2 to 0 by -1) {
      val z = zs(layer)
      val sp = sigmoidPrime(z)
      delta = (weights(layer + 1).t * delta) * sp
    }


    (nablaBiases, nablaWeights)

  }

  def feedForward(x: Vector): (List[Vector], List[Vector]) =
    biases.zip(weights).foldLeft(List.empty[Vector], List(x)) {
      case ((zs: List[Vector], as: List[Vector]), (b: Vector, w: Matrix)) =>
        val z: Vector = (w * as.last) + b
        val a: Vector = sigmoid(z)
        (zs :+ z, as :+ a)
    }

  def sigmoidPrime(z: Vector): Vector = {
    val sz = sigmoid(z)
    sz * (1.0 - sz)
  }

}

object NeuralNet {

  def oneHotEncoded(x: Int, base: Int = 10): DenseVector[Double] = {
    val v = DenseVector.zeros[Double](base)
    v(x) = 1.0
    v
  }

}
