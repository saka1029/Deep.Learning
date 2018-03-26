package deep.learning.test;

import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import deep.learning.common.Functions;
import deep.learning.common.INDArrayFunction;
import deep.learning.common.Util;

public class TestFunctions {

    @Test
    public void testCross_entropy_error() {
        INDArray t = Nd4j.create(new double[] {0, 0, 1, 0, 0, 0, 0, 0, 0, 0});
        INDArray y = Nd4j.create(new double[] {0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0});
        Functions.cross_entropy_error(y, t);
        assertEquals(0.51082545709933802, Functions.cross_entropy_error(y, t), 5e-6);
        y = Nd4j.create(new double[] {0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0});
        assertEquals(2.3025840929945458, Functions.cross_entropy_error(y, t), 5e-6);
    }

    static boolean assertEqualsWithEps(INDArray expected, INDArray actual) {
        if (!expected.equalsWithEps(actual, 5e-2))
            fail(String.format("expected=%s actual=%s%n",
                Util.toStringExact(expected), Util.toStringExact(actual)));
        return true;
    }

    @Test
    public void testNumerical_gradient() {
        INDArrayFunction f = x -> x.mmul(x.transpose()).getDouble(0);
//        INDArrayFunction f = x -> {
//            double x0 = x.getDouble(0);
//            double x1 = x.getDouble(1);
//            double r = x0 * x0 + x1 * x1;
//            System.out.printf("f([%f, %f]) = %f%n", x0, x1, r);
//            return r;
//        };

        INDArray x0 = Nd4j.create(new double[] {3, 4});
        INDArray g0 = Functions.numerical_gradient(f, x0);
//        System.out.println("g0=" + Util.toStringExact(g0));
        assertEqualsWithEps(Nd4j.create(new double[] {6, 8}), g0);
        assertEqualsWithEps(Nd4j.create(new double[] {3, 4}), x0);

        INDArray x1 = Nd4j.create(new double[] {0, 2});
        INDArray g1 = Functions.numerical_gradient(f, x1);
        assertEqualsWithEps(Nd4j.create(new double[] {0, 4}), g1);
        assertEqualsWithEps(Nd4j.create(new double[] {0, 2}), x1);

        INDArray x2 = Nd4j.create(new double[] {3, 0});
        INDArray g2 = Functions.numerical_gradient(f, x2);
        assertEqualsWithEps(Nd4j.create(new double[] {6, 0}), g2);
        assertEqualsWithEps(Nd4j.create(new double[] {3, 0}), x2);
    }

    interface F2 {
        double apply(double[] a);
    }

    @Test
    public void testNumerical_gradient2() {
        double h = 5e-3F;
        F2 f = a -> a[0] * a[0] + a[1] * a[1];
        double[] x = {3, 4};
        double[] x1 = {x[0] + h, x[1]};
        double g1 = f.apply(x1);
        double[] x2 = {x[0] - h, x[1]};
        double g2 = f.apply(x2);
        double g = (g1 - g2) / (2 * h);
        assertEquals(x[0] * 2, g, 5e-8);
    }

    @Test
    public void testSoftmax() {
        // 1次元配列の場合
        INDArray a = Nd4j.create(new double[] {0.3, 2.9, 4.0});
        INDArray y = Functions.softmax(a);
        assertEqualsWithEps(Nd4j.create(new double[] {0.01821127, 0.24519181, 0.73659691}), y);
        assertEquals(1.0, y.sumNumber().doubleValue(), 5e-6);
        // 2次元配列の場合
        INDArray a2 = Nd4j.create(new double[][] {{0.3, 2.9, 4.0}, {0.3, 2.9, 4.0}});
        INDArray y2 = Functions.softmax(a2);
        assertEqualsWithEps(
            Nd4j.create(new double[][] {{0.01821127, 0.24519181, 0.73659691},
                {0.01821127, 0.24519181, 0.73659691}}),
            y2);
        assertEquals(1.0, y2.getRow(0).sumNumber().doubleValue(), 5e-6);
        assertEquals(1.0, y2.getRow(1).sumNumber().doubleValue(), 5e-6);
    }

    @Test
    public void testArgmax() {
        // 1次元配列の場合
        INDArray a = Nd4j.create(new double[] {0.3, 2.9, 4.0});
        INDArray y = Functions.argmax(a);
        assertEqualsWithEps(Nd4j.create(new double[] {2}), y);
        assertEquals(2.0, y.sumNumber().doubleValue(), 5e-6);
        // 2次元配列の場合
        INDArray a2 = Nd4j.create(new double[][] {{0.3, 2.9, 4.0}, {0.3, 2.9, 4.0}});
        INDArray y2 = Functions.argmax(a2);
        assertEqualsWithEps( Nd4j.create( new double[][] {{2}, {2}}), y2);
        assertEquals(2.0, y2.getRow(0).sumNumber().doubleValue(), 5e-6);
        assertEquals(2.0, y2.getRow(1).sumNumber().doubleValue(), 5e-6);
    }

    @Test
    public void testSigmoid() {
        // 1次元配列の場合
        INDArray A1 = Nd4j.create(new double[] {0.3, 0.7, 1.1});
        INDArray Z1 = Functions.sigmoid(A1);
        assertEqualsWithEps(Nd4j.create(new double[] {0.57444252, 0.66818777, 0.75026011}), Z1);
        // 2次元配列の場合
        INDArray A2 = Nd4j.create(new double[][] {{0.3, 0.7, 1.1}, {0.3, 0.7, 1.1}});
        INDArray Z2 = Functions.sigmoid(A2);
        assertEqualsWithEps(
            Nd4j.create(new double[][] {
                {0.57444252, 0.66818777, 0.75026011},
                {0.57444252, 0.66818777, 0.75026011}}), Z2);
    }

    @Test
    public void testAverage() {
        INDArray a = Nd4j.create(new double[][] {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
        assertEquals(5.0, Functions.average(a), 5e-6);
    }

    @Test
    public void testArrange() {
        INDArray a = Functions.arrange(5);
        assertEquals("[0.00,1.00,2.00,3.00,4.00,5.00]", Util.string(a));
    }

    @Test
    public void testLogspace() {
        INDArray t1 = Functions.logspace(2, 3, 10);
        assertEquals(
            "[100.00,129.15,166.81,215.44,278.26,"
            + "359.38,464.16,599.48,774.26,1,000.00]",
            Util.string(t1));
    }

}
