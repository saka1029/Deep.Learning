package deep.learning.test;

import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import deep.learning.common.Util;

public class TestND4J {

    @Test
    public void testINDArrayGt() {
        INDArray a = Nd4j.create(new double[] {-2, 0, 3});
        assertEquals("[0.00,0.00,1.00]", Util.string(a.gt(0)));
        assertEquals("[0.00,1.00,1.00]", Util.string(a.gte(0)));
    }

    @Test
    public void testTranspose() {
        INDArray a = Nd4j.create(new double[][] {{1, 2, 3}, {4, 5, 6}});
        assertArrayEquals(new int[] {2, 3}, a.shape());
        assertArrayEquals(new int[] {3, 2}, a.transpose().shape());
        assertEquals("[[1.00,4.00],[2.00,5.00],[3.00,6.00]]", Util.string(a.transpose()));
        INDArray b = Nd4j.create(new double[] {1, 2, 3, 4});
        assertArrayEquals(new int[] {1, 4}, b.shape());
        assertArrayEquals(new int[] {4, 1}, b.transpose().shape());
        assertEquals("[1.00,2.00,3.00,4.00]", Util.string(b.transpose()));
    }

    @Test
    public void testSum() {
        INDArray a = Nd4j.create(new double[][] {{1, 2, 3}, {4, 5, 6}});
        assertEquals("[[1.00,2.00,3.00],[4.00,5.00,6.00]]", Util.string(a));
        assertArrayEquals(new int[] {1, 3}, a.sum(0).shape());
        assertEquals("[5.00,7.00,9.00]", Util.string(a.sum(0)));
        assertArrayEquals(new int[] {2, 1}, a.sum(1).shape());
        assertEquals("[6.00,15.00]", Util.string(a.sum(1)));
    }

    @Test
    public void test2D1D() {
        INDArray matrix = Nd4j.create(new double[][] {{1, 2}, {3, 4}});
        int size = matrix.size(0);
        for (int i = 0; i < size; ++i)
            matrix.putRow(i, matrix.getRow(i).mul(10));
        assertEquals("[[10.00,20.00],[30.00,40.00]]", Util.string(matrix));
    }

    @Test
    public void testConcat() {
        INDArray a = Nd4j.create(new double[] {1});
        INDArray b = Nd4j.create(new double[] {2});
        INDArray c = Nd4j.concat(1, a, b);
        assertEquals("[1.00,2.00]", Util.string(c));
    }

    @Test
    public void testArange() {
        INDArray a = Nd4j.arange(0, 3).div(100);
        assertEquals("[0.00,0.01,0.02]", Util.string(a));
        INDArray b = Nd4j.linspace(0.00, 0.02, 3);
        assertEquals("[0.00,0.01,0.02]", Util.string(b));
    }

    @Test
    public void testRepeat() {
        INDArray a = Nd4j.create(new double[] {0, 1, 2});
        INDArray b = Nd4j.create(new double[] {0, 4});
        assertEquals("[[0.00,1.00,2.00],[0.00,1.00,2.00]]", Util.string(Nd4j.repeat(a, b.size(1))));
        assertEquals("[[0.00,0.00,0.00],[4.00,4.00,4.00]]", Util.string(Nd4j.repeat(b, a.size(1)).transpose()));
    }

}
