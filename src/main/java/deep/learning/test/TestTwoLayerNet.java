package deep.learning.test;

import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import deep.learning.common.TwoLayerNet;

public class TestTwoLayerNet {

    @Test
    public void testTwoLayerNet() throws Exception {
        TwoLayerNet net = new TwoLayerNet(784, 100, 10);
        assertArrayEquals(new int[] {784, 100}, net.parms.W1.shape());
        assertArrayEquals(new int[] {1, 100}, net.parms.b1.shape());
        assertArrayEquals(new int[] {100, 10}, net.parms.W2.shape());
        assertArrayEquals(new int[] {1, 10}, net.parms.b2.shape());
    }

    @Test
    public void testAccuracy() throws Exception {
        TwoLayerNet net = new TwoLayerNet(
            Nd4j.create(new double[][] {{1, 1}, {1, 1}}),
            Nd4j.create(new double[] {1, 1}),
            Nd4j.create(new double[][] {{1, 1}, {1, 1}}),
            Nd4j.create(new double[] {1, 1}));
        INDArray x = Nd4j.create(new double[][] {{1, 0}, {0, 1}});
        INDArray t = Nd4j.create(new double[][] {{1, 0}, {0, 1}});
        double accuracy = net.accuracy(x, t);
        assertEquals(0.5, accuracy, 5e-6);
    }

}
