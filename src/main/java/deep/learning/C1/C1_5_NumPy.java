package deep.learning.C1;

import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import deep.learning.common.Util;

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
        // INDArrayのメソッドmul(INDArray)やadd(INDArray)は自動的にブロードキャストしません。
        // broadcast(int[])を使用して左辺の次元に合わせてやる必要があります。
        assertEquals("[[10.00,40.00],[30.00,80.00]]", Util.string(A.mul(B.broadcast(A.shape()))));
        // 単純に掛け算するとIllegalStateException: Mis matched shapesとなります。
        try {
            assertEquals("[[10.00,40.00],[30.00,80.00]]", Util.string(A.mul(B)));
            fail();
        } catch (IllegalStateException e) {
            assertEquals("Mis matched shapes", e.getMessage());
        }
        // あるいはmmulRowVector(INDArray)を使うこともできます。
        assertEquals("[[10.00,40.00],[30.00,80.00]]", Util.string(A.mulRowVector(B)));
    }

    @Test
    public void C1_5_6_要素へのアクセス() {
        INDArray X = Nd4j.create(new double[][] {{51, 55}, {14, 19}, {0, 4}});
        assertEquals("[[51.00,55.00],[14.00,19.00],[0.00,4.00]]", Util.string(X));
        assertEquals("[51.00,55.00]", Util.string(X.getRow(0)));
        assertEquals(55.0, X.getDouble(0, 1), 5e-6);
        // INDArrayはIterableインタフェースを実装していません。
        for (int i = 0, size = X.size(0); i < size; ++i)
            assertEquals(2, X.getRow(i).size(1));
        // Xをベクトルに変換します。
        X = Nd4j.toFlattened(X);
        assertEquals("[51.00,55.00,14.00,19.00,0.00,4.00]", Util.string(X));
        assertEquals("[51.00,14.00,0.00]", Util.string(X.getColumns(0, 2, 4)));
    }

    /**
     * <a href="https://nd4j.org/ja/userguide">データ型の設定</a>には以下の記述があります。
     * <blockquote>
     * データ型の設定<br>
     * ND4Jは現在、double精度値またはdouble精度値によるINDArrayによるバッキングを許可しています。
     * デフォルトは単精度（double）です。ND4Jがdouble精度に配列全体に使用する順序を設定するには、
     * 以下を使用することができます。
     * 0.4-rc3.8、及びそれ以前の場合、
     * <pre><code>
     * Nd4j.dtype = DataBuffer.Type.DOUBLE;
     * NDArrayFactory factory = Nd4j.factory();
     * factory.setDType(DataBuffer.Type.DOUBLE);
     * </code></pre>
     * 0.4-rc3.9、及びそれ以降の場合、
     * <pre><code>
     * DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
     * </code></pre>
     * </blockquote>
     */
    @Test
    public void データ型の設定() {
        // デフォルト精度の配列を作成します。
        INDArray a = Nd4j.create(new double[] {1D / 3});
        // 倍精度(double)に設定します。
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.DOUBLE);
        // 倍精度の配列を作成します。
        INDArray b = Nd4j.create(new double[] {1D / 3});
        // 変更されていることがわかります。
        assertEquals(DataBuffer.Type.DOUBLE, DataTypeUtil.getDtypeFromContext());
        // aはdoubleで初期化しましたが、単精度の配列です。
        assertEquals(0.3333333432674408, a.getDouble(0), 5e-14);
        // bはdoubleで初期化しましたが、倍精度の配列です。
        assertEquals(0.3333333333333333, b.getDouble(0), 5e-14);
        // 単精度(double)に戻します。
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.FLOAT);
    }

}
