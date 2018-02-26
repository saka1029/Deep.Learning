package deep.learning.C2;

import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import deep.learning.common.Util;

public class C2_3_パーセプトロンの実装 {

    public static int AND(int x1, int x2) {
        double w1 = 0.5, w2 = 0.5, theta = 0.7;
        double tmp = x1 * w1 + x2 * w2;
        if (tmp <= theta)
            return 0;
        else
            return 1;
    }

    @Test
    public void C2_3_1_簡単な実装() {
        assertEquals(0, AND(0, 0));
        assertEquals(0, AND(1, 0));
        assertEquals(0, AND(0, 1));
        assertEquals(1, AND(1, 1));
    }

    @Test
    public void C2_3_2_重みとバイアスの導入() {
        INDArray x = Nd4j.create(new double[] {0, 1});
        INDArray w = Nd4j.create(new double[] {0.5, 0.5});
        double b = -0.7;
        assertEquals("[0.00,0.50]", Util.string(w.mul(x)));
        assertEquals(0.5, w.mul(x).sumNumber().doubleValue(), 0.000005);
        assertEquals(-0.19999999999999996, w.mul(x).sumNumber().doubleValue() + b, 0.000005);
    }

    public static int AND2(int x1, int x2) {
        INDArray x = Nd4j.create(new double[] {x1, x2});
        INDArray w = Nd4j.create(new double[] {0.5, 0.5});
        double b = -0.7;
        double tmp = w.mul(x).sumNumber().doubleValue() + b;
        return tmp <= 0 ? 0 : 1;
    }

    public static int NAND(int x1, int x2) {
        INDArray x = Nd4j.create(new double[] {x1, x2});
        INDArray w = Nd4j.create(new double[] {-0.5, -0.5});
        double b = 0.7;
        double tmp = w.mul(x).sumNumber().doubleValue() + b;
        return tmp <= 0 ? 0 : 1;
    }

    public static int OR(int x1, int x2) {
        INDArray x = Nd4j.create(new double [] {x1, x2});
        INDArray w = Nd4j.create(new double[] {0.5, 0.5});
        double b = -0.2;
        double tmp = w.mul(x).sumNumber().doubleValue() + b;
        return tmp <= 0 ? 0 : 1;
    }

    @Test
    public void C2_3_3_重みとバイアスによる実装() {
        assertEquals(0, AND2(0, 0));
        assertEquals(0, AND2(1, 0));
        assertEquals(0, AND2(0, 1));
        assertEquals(1, AND2(1, 1));
        assertEquals(1, NAND(0, 0));
        assertEquals(1, NAND(1, 0));
        assertEquals(1, NAND(0, 1));
        assertEquals(0, NAND(1, 1));
        assertEquals(0, OR(0, 0));
        assertEquals(1, OR(1, 0));
        assertEquals(1, OR(0, 1));
        assertEquals(1, OR(1, 1));
    }

}
