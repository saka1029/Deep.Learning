package deep.learning.C4;

import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

import deep.learning.common.Functions;
import deep.learning.common.INDArrayFunction;
import deep.learning.common.Util;

public class C4_4_勾配 {

    public double function_2(INDArray x) {
        double x0 = x.getFloat(0);
        double x1 = x.getFloat(1);
        return x0 * x0 + x1 * x1;
        // または
        // return x.mul(x).sumNumber().doubleValue();
        // あるいは以下のように転置行列と内積をとることもできます。
        // return x.mmul(x.transpose()).getDouble(0);
    }

    @Test
    public void C4_4() {
        // 4.4節の冒頭のサンプルプログラムです。
        assertEquals("[5.99,8.00]", Util.string(Functions.numerical_gradient(this::function_2, Nd4j.create(new double[] {3.0, 4.0}))));
        assertEquals("[0.00,4.00]", Util.string(Functions.numerical_gradient(this::function_2, Nd4j.create(new double[] {0.0, 2.0}))));
        assertEquals("[5.99,0.00]", Util.string(Functions.numerical_gradient(this::function_2, Nd4j.create(new double[] {3.0, 0.0}))));
    }

    public static INDArray gradient_descent(INDArrayFunction f, INDArray init_x, double lr, int step_num) {
        INDArray x = init_x;
        for (int i = 0; i < step_num; ++i) {
            INDArray grad = Functions.numerical_gradient(f, x);
            INDArray y = x.sub(grad.mul(lr));
//            System.out.printf("step:%d x=%s grad=%s x'=%s%n", i, x, grad, y);
            x = y;
        }
        return x;
    }

    @Test
    public void C4_4_1_勾配法() {
    // lr = 0.1
    INDArray init_x = Nd4j.create(new double[] {-3.0, 4.0});
    INDArray r = gradient_descent(this::function_2, init_x, 0.1, 100);
    assertEquals("[-0.00,0.00]", Util.string(r));
    assertEquals(-6.11110793e-10, r.getDouble(0), 5e-6);
    assertEquals(8.14814391e-10, r.getDouble(1), 5e-6);
    // 学習率が大きすぎる例: lr = 10.0
    r = gradient_descent(this::function_2, init_x, 10.0, 100);
    // Pythonの結果とは同じになりませんが、いずれにしても正しい結果は得られません。
    assertEquals("[25,111.97,-33,524.69]", Util.string(r));
    // 学習率が小さすぎる例: lr = 1e-10
    r = gradient_descent(this::function_2, init_x, 1e-10, 100);
    assertEquals("[-3.00,4.00]", Util.string(r));
}

static class simpleNet {

    /** 重み */
    public final INDArray W;

    /**
     * 重みを0.0から1.0の範囲の乱数で初期化します。
     */
    public simpleNet() {
        try (Random r = new DefaultRandom()) {
            // 2x3のガウス分布に基づく乱数の行列を作成します。
            W = r.nextGaussian(new int[] {2, 3});
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * 本書と結果が一致することを確認するため重みを
     * 外部から与えることができるようにします。
     */
    public simpleNet(INDArray W) {
        this.W = W.dup();   // 防衛的にコピーします。
    }

    public INDArray predict(INDArray x) {
        return x.mmul(W);
    }

    public double loss(INDArray x, INDArray t) {
        INDArray z = predict(x);
        INDArray y = Functions.softmax(z);
        double loss = Functions.cross_entropy_error(y, t);
        return loss;
    }
}

@Test
public void C4_4_2_ニューラルネットワークに対する勾配() {
    // 重みは乱数ではなく本書と同じ値を与えてみます。
    INDArray W = Nd4j.create(new double[][] {
        {0.47355232, 0.9977393, 0.84668094},
        {0.85557411, 0.03563661, 0.69422093},
    });
    simpleNet net = new simpleNet(W);
    assertEquals("[[0.47,1.00,0.85],[0.86,0.04,0.69]]", Util.string(net.W));
    INDArray x = Nd4j.create(new double[] {0.6, 0.9});
    INDArray p = net.predict(x);
    assertEquals("[1.05,0.63,1.13]", Util.string(p));
    assertEquals(2, Functions.argmax(p).getInt(0));
    INDArray t = Nd4j.create(new double[] {0, 0, 1});
    assertEquals(0.92806853663411326, net.loss(x, t), 5e-6);
    // 関数定義はラムダ式を使っています。
    INDArrayFunction f = dummy -> net.loss(x, t);
    INDArray dW = Functions.numerical_gradient(f, net.W);
    assertEquals("[[0.22,0.14,-0.36],[0.33,0.22,-0.54]]", Util.string(dW));
}

/**
 * @see <a href="https://nd4j.org/doc/org/nd4j/linalg/api/rng/Random.html#nextGaussian--">nextGaussian</a>
 */
@Test
public void testRundomGaussian() throws Exception {
    // ND4JにおけるRandom.nextGaussian()の特性について調べます。
    // ドキュメントでは以下の用の記述されています。
    // Returns the next pseudorandom,
    // Gaussian ("normally") distributed double value with mean 0.0
    // and standard deviation 1.0 from this random number generator's sequence.
    // NumPyのrandnは「標準正規分布（ガウス分布）でランダムな数値を出力する。
    // 標準正規分布（ガウス分布）は、平均0, 標準偏差1の正規分布である。」なので
    // 同じ仕様であると考えられます。
    //
    // RandomインタフェースはAutoCloseableを実装しているので、
    // 最後にclose()を呼び出す必要があります。
    try (Random r = new DefaultRandom()) {
        int size = 1000;
        INDArray a = r.nextGaussian(new int[] {size});
        double min = a.minNumber().doubleValue();
        double max = a.maxNumber().doubleValue();
        double ave = a.sum(1).getDouble(0) / size;
        double std = a.std(1).getDouble(0);
//            System.out.printf("最小=%f 最大=%f 平均=%f 標準偏差=%f%n",
//                min, max, ave, std);
        // 平均はおおよそ0.0です。
        assertEquals(0.0, ave, 0.2);
        // 標準偏差はおよそ1.0です。
        assertEquals(1.0, std, 0.2);
    }
}

@Test
public void testArgmax() {
    INDArray a = Nd4j.create(new double[] {1, 2, 3});
    // 1次元の行列は先頭の添え字を省略してもアクセスできます。
    assertEquals(3, a.getInt(2));
    assertEquals(3, a.getInt(0, 2));
    INDArray m = Nd4j.create(new double[][] {{1, 2, 3}, {6, 5, 4}});
    // argmaxは各行における最大インデックスを求めます。
    INDArray x = Functions.argmax(m);
    assertEquals("[2.00,0.00]", Util.string(x));
    // 2x3行列のargmaxは2x1行列となります。
    // 一般にMxN行列のargmaxはMx1行列です。
    assertArrayEquals(new int[] {2, 1}, x.shape());
    // getInt(row, col)の「,col」は省略できます。
    assertEquals(2, x.getInt(0));
    assertEquals(0, x.getInt(1));
}

}
