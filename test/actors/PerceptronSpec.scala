package actors

import math.Vector
import akka.actor.ActorSystem
import akka.testkit.{ImplicitSender, TestKit, TestProbe}
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpecLike}

class PerceptronSpec extends TestKit(ActorSystem("PerceptronSpec")) with ImplicitSender
  with WordSpecLike with Matchers with BeforeAndAfterAll {

  override def afterAll {
    TestKit.shutdownActorSystem(system)
  }

  "A Perceptron Actor" must {
    "produce a NAND gate" in {
      val weights = Vector(-2.0, -2.0)
      val bias = 3.0
      val targets = List(self)

      val perceptron = system.actorOf(Perceptron.props(weights, bias, targets))

      perceptron ! 0.0
      perceptron ! 0.0
      expectMsg(1.0)

      perceptron ! 1.0
      perceptron ! 0.0
      expectMsg(1.0)

      perceptron ! 0.0
      perceptron ! 1.0
      expectMsg(1.0)

      perceptron ! 1.0
      perceptron ! 1.0
      expectMsg(0.0)

    }

    "produce an ADDER" in {

      val sumProbe = TestProbe()
      val carryProbe = TestProbe()

      val weights = Vector(-2.0, -2.0)
      val bias = 3.0

      val n4 = system.actorOf(Perceptron.props(weights, bias, List(sumProbe.ref)))
      val n1 = system.actorOf(Perceptron.props(weights, bias, List(n4)))
      val n2 = system.actorOf(Perceptron.props(weights, bias, List(n4)))

      val n3 = system.actorOf(Perceptron.props(Vector(-4), bias, List(carryProbe.ref)))

      val n0 = system.actorOf(Perceptron.props(weights, bias, List(n1, n2, n3)))

      def add(x1: Double, x2: Double): Unit = {
        n0 ! x1
        n1 ! x1

        n0 ! x2
        n2 ! x2
      }

      add(0.0, 0.0)
      sumProbe.expectMsg(0.0)
      carryProbe.expectMsg(0.0)

      add(0.0, 1.0)
      sumProbe.expectMsg(1.0)
      carryProbe.expectMsg(0.0)

      add(1.0, 0.0)
      sumProbe.expectMsg(1.0)
      carryProbe.expectMsg(0.0)

      add(1.0, 1.0)
      sumProbe.expectMsg(0.0)
      carryProbe.expectMsg(1.0)

    }
  }

}
