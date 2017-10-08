package nn

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j.randn
import org.nd4j.linalg.ops.transforms.Transforms._
import org.nd4s.Implicits._

import scala.collection.parallel.ParSeq
import scala.language.postfixOps

object PerfTest extends App {

  val t0 = System.currentTimeMillis()

  val topology = List(786, 30, 10)

  val biases: ParSeq[INDArray] =
    topology.tail.map(size => randn(size, 1)).par

  val weights: ParSeq[INDArray] =
    topology.sliding(2).map(t => randn(t(1), t.head)).toList.par

  (1 to 100000).foreach { i =>
    val x = randn(786, 1)
    val activations: ParSeq[INDArray] = biases.zip(weights).foldLeft(ParSeq(x)) {
      case (as, (b, w)) =>
        val z = (w dot as.last) + b
        val a = sigmoid(z)
        as :+ a
    }
  }

  val t1 = System.currentTimeMillis()

  println(t1 - t0)

}
