package nn

import org.nd4j.jita.conf.CudaEnvironment
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j._
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4j.linalg.api.buffer.DataBuffer
import org.nd4j.linalg.api.buffer.util.DataTypeUtil

import scala.util.Random

class Nd4jNet(topology: List[Int]) {

  CudaEnvironment
    .getInstance()
    .getConfiguration
//    .setPoolSize(24)
    .setVerbose(true)
//    .enableDebug(true)

  DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE)

  val biases: List[INDArray] =
    topology.tail.map(size => randn(size, 1))

  val weights: List[INDArray] =
    topology.sliding(2).map(t => randn(t(1), t.head)).toList

  /**
    * Train the neural network using mini-batch stochastic
    * gradient descent.  The ``training_data`` is a list of tuples
    * ``(x, y)`` representing the training inputs and the desired
    * outputs.
    */
  def sgd(trainingData: List[(INDArray, INDArray)],
          epochs: Int,
          miniBatchSize: Int,
          learningRate: Double): Unit = {

    (1 to epochs).foreach { j =>
      val shuffled = Random.shuffle(trainingData)
      shuffled.sliding(miniBatchSize).foreach { miniBatch =>
        updateMiniBatch(miniBatch, learningRate)
      }
      println(s"Epoch $j complete")
    }
  }

  def feedForward(x: INDArray): (List[INDArray], List[INDArray]) = {
    biases.zip(weights).foldLeft(List.empty[INDArray], List(x)) {
      case ((zs, as), (b, w)) =>
        // z = np.dot(w, activation)+b
        val z = w.mmul(as.last).add(b)
        val a = sigmoid(z)
        (zs :+ z, as :+ a)
    }
  }

  /**
    * Update the network's weights and biases by applying
    * gradient descent using backpropagation to a single mini batch.
    * The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
    * is the learning rate
    */
  def updateMiniBatch(miniBatch: List[(INDArray, INDArray)],
                      learningRate: Double): Unit = {

    val nablaBiases = biases.map(b => zeros(b.shape(): _*))
    val nablaWeights = weights.map(w => zeros(w.shape(): _*))

    miniBatch.foreach {
      case (x, y) =>
        val (deltaNablaB, deltaNablaW) = backProp(x, y)

        nablaBiases.zip(deltaNablaB).foreach {
          case (nb, dnb) =>
            nb.addi(dnb)
        }

        nablaWeights.zip(deltaNablaW).foreach {
          case (nw, dnw) =>
            nw.addi(dnw)
        }
    }

    biases.zip(nablaBiases).foreach {
      case (b, nb) =>
        b.subi(nb.mul(learningRate * miniBatch.size))
    }

    weights.zip(nablaWeights).foreach {
      case (w, nw) =>
        w.subi(nw.mul(learningRate * miniBatch.size))
    }

  }

  /**
    * Return a tuple ``(nabla_b, nabla_w)`` representing the
    * gradient for the cost function C_x.  ``nabla_b`` and
    * ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
    * to ``self.biases`` and ``self.weights``.
    */
  def backProp(x: INDArray, y: INDArray): (List[INDArray], List[INDArray]) = {

    val (zs, activations) = feedForward(x)

    // delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
    // delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
    val a = activations.last.sub(y)
    val sp = sigmoidPrime(zs.last)
    val delta = a.mul(sp)

    val inb = delta
    // np.dot(delta, activations[-2].transpose())
    val inw = delta.mmul(activations(activations.size - 2).transpose())

    val (nablaBiases, nablaWeights) = (2 until topology.size)
      .foldLeft((List(inb), List(inw))) {
        case ((nbl, nwl), l) =>
          // z = zs[-l]
          val z = zs(zs.size - l)
          val sp = sigmoidPrime(z)

          // delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
          val delta = weights(weights.size - l + 1)
            .transpose()
            .mmul(nbl.head) // last added nbl is the previous delta
            .mul(sp)

          val nb = delta
          // nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
          val nw =
            delta.mmul(activations(activations.size - l - 1).transpose())

          (nb :: nbl, nw :: nwl)
      }

    (nablaBiases, nablaWeights)
  }

  /**
    * Derivative of the sigmoid function.
    */
  def sigmoidPrime(z: INDArray): INDArray = {
    // sigmoid(z)*(1-sigmoid(z))
    val sz = sigmoid(z)
    sz.mul(sz.neg().add(1.0))
  }

}

object Nd4jNet {

  def oneHotEncoded(x: Int, base: Int = 10): INDArray = {
    val v = zeros(1, base)
    v.putScalar(x, 1.0)
  }
}
