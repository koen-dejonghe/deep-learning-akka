package actors

import akka.actor.ActorSystem
import akka.testkit.{ImplicitSender, TestKit, TestProbe}
import math.Vector
import org.scalatest.{BeforeAndAfterAll, Matchers, WordSpecLike}

class NetworkSpec extends TestKit(ActorSystem("NetworkSpec")) with ImplicitSender
  with WordSpecLike with Matchers with BeforeAndAfterAll {

  override def afterAll {
    TestKit.shutdownActorSystem(system)
  }

  "A Network Actor" must {
    "forward propagate" in {

      val probe = TestProbe()

      val topology = List(3, 15, 10)
      val learningRate = 3.0
      val network = system.actorOf(Network.props(topology, learningRate, probe.ref))

      val v = Vector.randn(3)
      network ! (v, Vector.oneHotEncoded(9))

      val output = probe.receiveN(10)
      println(output)

      probe.expectNoMsg()
    }

    "forward propagate again" in {
      val probe = TestProbe()

      val topology = List(1, 1)
      val learningRate = 3.0
      val network = system.actorOf(Network.props(topology, learningRate, probe.ref))
      network ! Vector(0.1)
      val output = probe.receiveN(1).head.asInstanceOf[Activation]
      println(output)
      network ! Vector(output.value)
      val output2 = probe.receiveN(1).head.asInstanceOf[Activation]
      println(output2)
    }
  }

}
