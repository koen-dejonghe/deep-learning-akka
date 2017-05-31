package actors

import math.Vector
import akka.actor.{Actor, ActorRef, Props}

import scala.language.postfixOps

class Perceptron(weights: Vector, bias: Double, targets: List[ActorRef]) extends Actor {

  override def receive: Receive = collect(Vector())

  def collect(v: Vector): Receive = {
    case x: Double =>
      val xs = v :+ x
      if (xs.size == weights.size){
        fire(xs)
        context.become(collect(Vector()))
      } else {
        context.become(collect(xs))
      }
    case _ =>
  }

  def fire(xs: Vector): Unit =
    if (xs.dot(weights) + bias > 0.0)
      targets.foreach(t => t ! 1.0)
    else
      targets.foreach(t => t ! 0.0)

}

object Perceptron {
  def props(weights: Vector, bias: Double, targets: List[ActorRef]): Props =
    Props(new Perceptron(weights, bias, targets))

}
