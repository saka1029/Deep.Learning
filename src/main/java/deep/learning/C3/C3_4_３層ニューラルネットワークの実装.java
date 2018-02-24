package deep.learning.C3;

import static org.junit.Assert.*;

import java.util.HashMap;
import java.util.Map;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import deep.learning.common.Util;

public class C3_4_３層ニューラルネットワークの実装 {

    @Test
    public void C3_4_2_各層における信号伝達の実装() {
        INDArray X = Nd4j.create(new double[] {1.0, 0.5});
        INDArray W1 = Nd4j.create(new double[][] {{0.1, 0.3, 0.5}, {0.2, 0.4, 0.6}});
        INDArray B1 = Nd4j.create(new double[] {0.1, 0.2, 0.3});
        assertArrayEquals(new int[] {2, 3}, W1.shape());
        assertArrayEquals(new int[] {1, 2}, X.shape());
        assertArrayEquals(new int[] {1, 3}, B1.shape());
        INDArray A1 = X.mmul(W1).add(B1);
        INDArray Z1 = Transforms.sigmoid(A1);
        assertEquals("[0.30,0.70,1.10]", Util.string(A1));
        assertEquals("[0.57,0.67,0.75]", Util.string(Z1));

        INDArray W2 = Nd4j.create(new double[][] {{0.1, 0.4}, {0.2, 0.5}, {0.3, 0.6}});
        INDArray B2 = Nd4j.create(new double[] {0.1, 0.2});
        assertArrayEquals(new int[] {1, 3}, Z1.shape());
        assertArrayEquals(new int[] {3, 2}, W2.shape());
        assertArrayEquals(new int[] {1, 2}, B2.shape());
        INDArray A2 = Z1.mmul(W2).add(B2);
        INDArray Z2 = Transforms.sigmoid(A2);
        assertEquals("[0.52,1.21]", Util.string(A2));
        assertEquals("[0.63,0.77]", Util.string(Z2));

        INDArray W3 = Nd4j.create(new double[][] {{0.1, 0.3}, {0.2, 0.4}});
        INDArray B3 = Nd4j.create(new double[] {0.1, 0.2});
        INDArray A3 = Z2.mmul(W3).add(B3);
        // ND4JにはTransformsクラスにidentity(INDArray)メソッドが用意されています。
        INDArray Y = Transforms.identity(A3);
        assertEquals("[0.32,0.70]", Util.string(A3));
        assertEquals("[0.32,0.70]", Util.string(Y));
        // Y.equals(A3)は真となります。
        assertEquals(A3, Y);
    }

    public static Map<String, INDArray> init_network() {
        Map<String, INDArray> network = new HashMap<>();
        network.put("W1", Nd4j.create(new double[][] {{0.1, 0.3, 0.5}, {0.2, 0.4, 0.6}}));
        network.put("b1", Nd4j.create(new double[] {0.1, 0.2, 0.3}));
        network.put("W2", Nd4j.create(new double[][] {{0.1, 0.4}, {0.2, 0.5}, {0.3, 0.6}}));
        network.put("b2", Nd4j.create(new double[] {0.1, 0.2}));
        network.put("W3", Nd4j.create(new double[][] {{0.1, 0.3}, {0.2, 0.4}}));
        network.put("b3", Nd4j.create(new double[] {0.1, 0.2}));
        return network;
    }

    public static INDArray forward(Map<String, INDArray> network, INDArray x) {
        INDArray W1 = network.get("W1");
        INDArray W2 = network.get("W2");
        INDArray W3 = network.get("W3");
        INDArray b1 = network.get("b1");
        INDArray b2 = network.get("b2");
        INDArray b3 = network.get("b3");

        INDArray a1 = x.mmul(W1).add(b1);
        INDArray z1 = Transforms.sigmoid(a1);
        INDArray a2 = z1.mmul(W2).add(b2);
        INDArray z2 = Transforms.sigmoid(a2);
        INDArray a3 = z2.mmul(W3).add(b3);
        INDArray y = Transforms.identity(a3);
        return y;
    }

    @Test
    public void C3_4_3_実装のまとめ() {
        Map<String, INDArray> network = init_network();
        INDArray x = Nd4j.create(new double[] {1.0, 0.5});
        INDArray y = forward(network, x);
        assertEquals("[0.32,0.70]", Util.string(y));
    }

}
