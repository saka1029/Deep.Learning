package deep.learning.C5;

import static org.junit.Assert.*;

import java.util.function.DoubleUnaryOperator;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import deep.learning.common.Affine;
import deep.learning.common.Functions;
import deep.learning.common.Relu;
import deep.learning.common.Sigmoid;
import deep.learning.common.Util;

public class TestLayers {

    @Test
    public void testRelu() {
        INDArray x = Nd4j.create(new double[][] {{1.0, -0.5}, {-2.0, 3.0}});
        assertEquals("[[1.00,-0.50],[-2.00,3.00]]", Util.string(x));
        Relu relu = new Relu();
        INDArray a = relu.forward(x);
        // forwardの結果
        assertEquals("[[1.00,0.00],[0.00,3.00]]", Util.string(a));
        // mask
        assertEquals("[[1.00,0.00],[0.00,1.00]]", Util.string(relu.mask));
        INDArray dout = Nd4j.create(new double[][] {{5, 6}, {7, 8}});
        INDArray b = relu.backward(dout);
        // backwardの結果
        assertEquals("[[5.00,0.00],[0.00,8.00]]", Util.string(b));
        DoubleUnaryOperator f = p -> p > 0.0 ? p : 0.0;
        // 数値微分との比較
        INDArray n = Functions.numerical_gradient(f, x);
        assertEquals(Util.string(n.mul(dout)), Util.string(b));
    }

    @Test
    public void testSigmoid() {
        Sigmoid layer = new Sigmoid();
        INDArray x = Nd4j.create(new double[][] {{1, 2},{3, 4}});
        INDArray y = layer.forward(x);
        assertEquals(Util.string(Transforms.exp(x.neg()).add(1.0).rdiv(1.0)),
            Util.string(y));
        INDArray dout = Nd4j.create(new double[][] {{5, 6}, {7, 8}});
        INDArray b = layer.backward(dout);
        assertEquals(Util.string(y.mul(y.rsub(1.0).mul(dout))), Util.string(b));
        // 数値微分との比較
        DoubleUnaryOperator f = a -> 1.0 / (1.0 + Math.exp(-a));
        INDArray n = Functions.numerical_gradient(f, x);
        assertEquals(Util.string(n.mul(dout)), Util.string(b));
    }

    @Test
    public void testAffine() {
        INDArray W = Nd4j.create(new double[][] {{0.1, 0.2}, {0.3, 0.4}});
        INDArray b = Nd4j.zeros(2);
        Affine layer = new Affine(W, b);
        INDArray x = Nd4j.create(new double[] {-7, 2});
        INDArray y = layer.forward(x);
        assertEquals("[-0.10,-0.60]", Util.string(y));
        INDArray dout = Nd4j.create(new double[] {0.2, 0.3});
        INDArray z = layer.backward(dout);
        System.out.println("dW=" + layer.dW);
        System.out.println("db=" + layer.db);
//        assertEquals("[0.00,0.00]", Util.string(z));
    }

}
