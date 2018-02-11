package deep.learning.C3;

import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import test.deep.learning.Util;

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

}
