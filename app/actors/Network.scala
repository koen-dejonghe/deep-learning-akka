package actors

import akka.actor.{Actor, ActorRef, Props}
import math.Vector

import scala.language.postfixOps
import scala.util.Random

class Network(topology: List[Int], collector: ActorRef) extends Actor {

  val numLayers: Int = topology.size
  val network: List[List[ActorRef]] = build(numLayers-1, List(List(collector)))

  def build(i: Int, accumulator: List[List[ActorRef]]): List[List[ActorRef]] = {
    i match {
      case 0 => accumulator
      case _ =>
        val layer = buildLayer(topology(i), topology(i-1), accumulator.head)
        build(i-1, layer :: accumulator)
    }
  }

  def buildLayer(size: Int,
                 sizePrevLayer: Int,
                 targetLayer: List[ActorRef]): List[ActorRef] = {
    val seq = for (i <- 0 until size) yield {
      val weights = Vector.randn(sizePrevLayer)
      val bias = Random.nextGaussian()
      context.system.actorOf(Neuron.props(targetLayer, i, weights, bias, sigmoid))
    }
    seq.toList
  }

  def sigmoid(z: Double): Double = 1 / (1 + Math.exp(-z))

  override def receive: Receive = {
    case v: Vector =>
      require(v.size == topology.head)
      v.xs.zipWithIndex.foreach { x =>
        network.head.foreach { node =>
          node ! Output(x._2, x._1)
        }
      }
    case _ =>
  }
}

object Network {
  def props(topology: List[Int], collector: ActorRef): Props =
    Props(new Network(topology, collector))
}
