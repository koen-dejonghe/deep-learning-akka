package actors

import akka.actor.{Actor, ActorLogging, ActorRef, Props}
import math._

import scala.annotation.tailrec
import scala.language.postfixOps
import scala.util.Random

class Network(topology: List[Int], learningRate: Double, collector: ActorRef) extends Actor with ActorLogging {

  val numLayers: Int = topology.size
  val network: List[List[ActorRef]] =
    build(numLayers - 1, List(List(self)))

  @tailrec
  private def build(i: Int,
                    accumulator: List[List[ActorRef]]): List[List[ActorRef]] =
    i match {
      case 0 => accumulator // do not include input layer
      case _ =>
        val layer = buildLayer(i, topology(i), topology(i - 1), accumulator.head)
        build(i - 1, layer :: accumulator)
    }

  def buildLayer(layer: Int,
                 layerSize: Int,
                 prevLayerSize: Int,
                 targetLayer: List[ActorRef]): List[ActorRef] = {
    val seq = for (neuron <- 0 until layerSize) yield {
      val weights = Vector.randn(prevLayerSize)
      val bias = Random.nextGaussian()
      context.system
        .actorOf(Neuron.props(targetLayer, neuron, learningRate, weights, bias, sigmoid))
    }
    seq.toList
  }

  override def receive: Receive = {
    case (x: Vector, y: Vector) =>

      log.debug(s"received: $x, $y")

      require(x.size == topology.head)
      x.output().foreach { o =>
        network.head.foreach { node =>
          node ! o
        }
      }
      context.become(sgd(x, y))

    case unknown =>
      log.debug(s"ignoring: $unknown")

  }

  def sgd(x: Vector, y: Vector, output: Map[Int, Double] = Map.empty[Int, Double]): Receive = {
    case Activation(n, activation) =>

      log.debug(s"sgd received activation: $n, $activation")
      val delta = (activation - y.xs(n)) * derivative(activation)

      sender() ! Delta(n, delta)

    case Delta(n, d) =>
      log.debug(s"sgd received delta: ($n, $d)")

    case unknown =>
      log.debug(s"ignoring: $unknown")
  }

}

object Network {

  def props(topology: List[Int], learningRate: Double, collector: ActorRef): Props =
    Props(new Network(topology, learningRate, collector))

}
