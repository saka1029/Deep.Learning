package deep.learning.C1;

import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import test.deep.learning.Util;

/**
 * @see <a href="https://nd4j.org/doc/org/nd4j/linalg/factory/Nd4j.html">Nd4j</a>
 * @see <a href="https://nd4j.org/doc/org/nd4j/linalg/api/ndarray/INDArray.html">INDArray</a>
 */
public class C1_5_NumPy {

    @Test
    public void C1_5_2_NumPy配列の生成() {
        INDArray x = Nd4j.create(new double[] {1.0, 2.0, 3.0});
        assertEquals("[1.00,2.00,3.00]", Util.string(x));
    }

    @Test
    public void C1_5_3_NumPyの算術計算() {
        INDArray x = Nd4j.create(new double[] {1.0, 2.0, 3.0});
        INDArray y = Nd4j.create(new double[] {2.0, 4.0, 6.0});
        assertEquals("[3.00,6.00,9.00]", Util.string(x.add(y)));
        assertEquals("[-1.00,-2.00,-3.00]", Util.string(x.sub(y)));
        assertEquals("[2.00,8.00,18.00]", Util.string(x.mul(y)));
        assertEquals("[0.50,0.50,0.50]", Util.string(x.div(y)));
        // 1次元配列xのshapeは1行3列となります。
        assertArrayEquals(new int[] {1,3}, x.shape());
        assertEquals(2, x.rank());
    }

    @Test
    public void C1_5_4_NumPyのN次元配列() {
        INDArray A = Nd4j.create(new double[][] {{1, 2}, {3, 4}});
        assertEquals("[[1.00,2.00],[3.00,4.00]]", Util.string(A));
        assertArrayEquals(new int[] {2,2}, A.shape());
//        assertEquals("dtype('int64')", A.dtype());
        assertEquals(2, A.rank());
    }

    @Test
    public void C1_5_5_ブロードキャスト() {
        INDArray A = Nd4j.create(new double[][] {{1, 2}, {3, 4}});
        INDArray B = Nd4j.create(new double[] {10, 20});
        // broadcast(int[])を使用して左辺の次元に合わせてやる必要があります。
        assertEquals("[[10.00,40.00],[30.00,80.00]]", Util.string(A.mul(B.broadcast(A.shape()))));
        // 単純に掛け算するとIllegalStateException: Mis matched shapesとなります。
        try {
            assertEquals("[[10.00,40.00],[30.00,80.00]]", Util.string(A.mul(B)));
            fail();
        } catch (IllegalStateException e) {
            assertEquals("Mis matched shapes", e.getMessage());
        }
    }

    @Test
    public void C1_5_6_要素へのアクセス() {
        INDArray X = Nd4j.create(new double[][] {{51, 55}, {14, 19}, {0, 4}});
        assertEquals("[[51.00,55.00],[14.00,19.00],[0.00,4.00]]", Util.string(X));
        assertEquals("[51.00,55.00]", Util.string(X.getRow(0)));
        assertEquals(55.0, X.getDouble(0, 1), 0.000005);
        // INDArrayはIterableインタフェースを実装していません。
        for (int i = 0, size = X.size(0); i < size; ++i)
            System.out.println(X.getRow(i));
        X = Nd4j.toFlattened(X);    // Xを1次元の配列へ変換
        assertEquals("[51.00,55.00,14.00,19.00,0.00,4.00]", Util.string(X));
        assertEquals("[51.00,14.00,0.00]", Util.string(X.getColumns(0, 2, 4)));
    }

}
