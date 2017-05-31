package nn

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._

class Nd4jNet(topology: List[Int]) {

  /*
  CudaEnvironment
    .getInstance()
    .getConfiguration
    .setPoolSize(24)
    .setVerbose(true)
    .enableDebug(true)
    */

  val biases: List[INDArray] = topology.tail.map { size =>
    randn(size, 1)
  }

  val weights: List[INDArray] =
    topology.sliding(2).map(t => randn(t(1), t.head)).toList

  def feedForward(x: INDArray): (List[INDArray], List[INDArray]) = {
    biases.zip(weights).foldLeft(List.empty[INDArray], List(x)) {
      case ((zs, as), (b, w)) =>
        val z = w.mmul(as.last).add(b) // w * as[-1] + b
        val a = sigmoid(z)
        (zs :+ z, as :+ a)
    }
  }

  def backProp(x: INDArray, y: INDArray): (List[INDArray], List[INDArray]) = {

    val (zs, activations) = feedForward(x)

    // delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
    val a = activations.last.sub(y)
    val sp = sigmoidPrime(zs.last)
    var delta = a.mul(sp)

    val nablaBiases: List[INDArray] = topology.tail.map { size =>
      zeros(size, 1)
    }
    val nablaWeights: List[INDArray] =
      topology.sliding(2).map(t => zeros(t(1), t.head)).toList

    nablaBiases.last.setData(delta.data())
    nablaWeights.last.setData(
      delta.mmul(activations(activations.size - 2).transpose()).data())

    for (l <- 2 until topology.size) {

      val z = zs(zs.size - l)
      val sp = sigmoidPrime(z)

      delta = weights(weights.size - l + 1)
        .transpose()
        .mmul(delta)
        .mul(sp)

      nablaBiases(nablaBiases.size - l).setData(delta.data())
      nablaWeights(nablaWeights.size - l).setData(
        delta.mmul(activations(activations.size - l - 1).transpose()).data())
    }

    (nablaBiases, nablaWeights)
  }

  def sigmoidPrime(z: INDArray): INDArray = {
    val sz = sigmoid(z)
    sz.mul(sz.add(-1.0))
  }

}

object Nd4jNet {

  def oneHotEncoded(x: Int, base: Int = 10): INDArray = {
    val v = zeros(1, base)
    v.putScalar(x, 1.0)
  }
}
