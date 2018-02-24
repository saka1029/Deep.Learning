package deep.learning.C3;

import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import deep.learning.common.Util;

public class C3_3_多次元配列の計算 {

    @Test
    public void C3_3_1_多次元配列() {
        // 1次元の配列
        INDArray A = Nd4j.create(new double[] {1, 2, 3, 4});
        assertEquals("[1.00,2.00,3.00,4.00]", Util.string(A));
        // ND4Jでは1次元配列は1×Nの2次元配列となります。
        assertArrayEquals(new int[] {1, 4}, A.shape());
        // ND4Jでは次元数はrank()メソッドで求めます。
        assertEquals(2, A.rank());
        assertEquals(1, A.size(0));  // 行数
        assertEquals(4, A.size(1));  // 列数

        // 2次元の配列
        INDArray B = Nd4j.create(new double[][] {{1, 2}, {3, 4}, {5, 6}});
        assertEquals("[[1.00,2.00],[3.00,4.00],[5.00,6.00]]", Util.string(B));
        assertEquals(2, B.rank());
        assertArrayEquals(new int[] {3, 2}, B.shape());
    }

    @Test
    public void C3_3_2_行列の積() {
        INDArray A = Nd4j.create(new double[][] {{1, 2}, {3, 4}});
        assertArrayEquals(new int[] {2, 2}, A.shape());
        INDArray B = Nd4j.create(new double[][] {{5, 6}, {7, 8}});
        assertArrayEquals(new int[] {2, 2}, B.shape());
        // ND4Jの積はmmul(INDArray)メソッドで行います。
        assertEquals("[[19.00,22.00],[43.00,50.00]]", Util.string(A.mmul(B)));

        A = Nd4j.create(new double[][] {{1, 2, 3}, {4, 5, 6}});
        assertArrayEquals(new int[] {2, 3}, A.shape());
        B = Nd4j.create(new double[][] {{1, 2}, {3, 4}, {5, 6}});
        assertArrayEquals(new int[] {3, 2}, B.shape());
        assertEquals("[[22.00,28.00],[49.00,64.00]]", Util.string(A.mmul(B)));

        INDArray C = Nd4j.create(new double[][] {{1, 2}, {3, 4}});
        assertArrayEquals(new int[] {2, 2}, C.shape());
        assertArrayEquals(new int[] {2, 3}, A.shape());
        try {
            // ND4Jでは積をとる行列の要素数に誤りがある場合、
            // ND4JIllegalStateExceptionをスローします。
            A.mmul(C);
            fail();
        } catch (ND4JIllegalStateException e) {
            assertEquals(
                "Cannot execute matrix multiplication: [2, 3]x[2, 2]: "
                + "Column of left array 3 != rows of right 2"
                , e.getMessage());
        }

        A = Nd4j.create(new double[][] {{1, 2}, {3, 4}, {5, 6}});
        assertArrayEquals(new int[] {3, 2}, A.shape());
        B = Nd4j.create(new double[] {7, 8});
        assertArrayEquals(new int[] {1, 2}, B.shape());
        // ND4Jでは1次元配列は1×N行の行列となるため
        // 積を求める場合はtranspose()メソッドで転置する必要があります。
        assertArrayEquals(new int[] {2, 1}, B.transpose().shape());
        assertEquals("[23.00,53.00,83.00]", Util.string(A.mmul(B.transpose())));
    }

    @Test
    public void C3_3_3_ニューラルネットワークの行列の積() {
        INDArray X = Nd4j.create(new double[] {1, 2});
        assertArrayEquals(new int[] {1, 2}, X.shape());
        INDArray W = Nd4j.create(new double[][] {{1, 3, 5}, {2, 4, 6}});
        assertEquals("[[1.00,3.00,5.00],[2.00,4.00,6.00]]", Util.string(W));
        assertArrayEquals(new int[] {2, 3}, W.shape());
        INDArray Y = X.mmul(W);
        assertEquals("[5.00,11.00,17.00]", Util.string(Y));
    }

}
