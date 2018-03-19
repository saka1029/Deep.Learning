package deep.learning.C5;

import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

import deep.learning.common.Util;

public class C5_6_Affine_Softmaxレイヤの実装 {

    @Test
    public void C5_6_1_Affineレイヤ() throws Exception {
        try (Random r = new DefaultRandom()) {
            INDArray X = r.nextGaussian(new int[] {2});
            INDArray W = r.nextGaussian(new int[] {2, 3});
            INDArray B = r.nextGaussian(new int[] {3});
            assertArrayEquals(new int[] {1, 2}, X.shape());
            assertArrayEquals(new int[] {2, 3}, W.shape());
            assertArrayEquals(new int[] {1, 3}, B.shape());
            INDArray Y = X.mmul(W).addRowVector(B);
            assertArrayEquals(new int[] {1, 3}, Y.shape());
        }
    }

//    static class Affine {
//
//        final INDArray W, b;
//        INDArray x, dW, db;
//
//        public Affine(INDArray W, INDArray b) {
//            this.W = W;
//            this.b = b;
//        }
//
//        public INDArray forward(INDArray x) {
//            this.x = x;
//            return x.mmul(W).addRowVector(b);
//        }
//
//        public INDArray backward(INDArray dout) {
//            INDArray dx = dout.mmul(W.transpose());
//            this.dW = x.transpose().mmul(dout);
//            this.db = dout.sum(0);
//            return dx;
//        }
//    }

    @Test
    public void C5_6_2_バッチ版Affineレイヤ() throws Exception {
        INDArray X_dot_W = Nd4j.create(new double[][] {{0, 0, 0}, {10, 10, 10}});
        INDArray B = Nd4j.create(new double[] {1, 2, 3});
        assertEquals("[[0.00,0.00,0.00],[10.00,10.00,10.00]]", Util.string(X_dot_W));
        assertEquals("[[1.00,2.00,3.00],[11.00,12.00,13.00]]", Util.string(X_dot_W.addRowVector(B)));
        INDArray dY = Nd4j.create(new double[][] {{1, 2, 3}, {4, 5, 6}});
        assertEquals("[[1.00,2.00,3.00],[4.00,5.00,6.00]]", Util.string(dY));
        assertEquals("[5.00,7.00,9.00]", Util.string(dY.sum(0)));
        // TODO: バッチ版Affineレイヤのテスト
    }

//    static class SoftmaxWithLoss {
//
//        double loss;  // 損失
//        INDArray y;     // softmaxの出力
//        INDArray  t;    // 教師データ(one-hot vector)
//
//        public double forward(INDArray x, INDArray t) {
//            this.t = t;
//            this.y = Functions.softmax(x);
//            this.loss = Functions.cross_entropy_error(this.y, this.t);
//            return this.loss;
//        }
//
//        public INDArray backward(INDArray dout) {
//            int batch_size = this.t.size(0);
//            return y.sub(t).div(batch_size);
//        }
//    }

    @Test
    public void C5_6_3_Softmax_with_Lossレイヤ() {
        // TODO: SoftmaxWithLossのテスト
    }

}
