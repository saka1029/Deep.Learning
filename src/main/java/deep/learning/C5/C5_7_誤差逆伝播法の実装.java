package deep.learning.C5;

import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import deep.learning.common.Constants;
import deep.learning.common.Functions;
import deep.learning.common.MNISTImages;
import deep.learning.common.TwoLayerParams;

public class C5_7_誤差逆伝播法の実装 {

//    @Ignore
    @Test
    public void C5_7_3_誤差逆伝播法の勾配確認() throws Exception {
        // MNISTの訓練データを読み込みます。
        MNISTImages train = new MNISTImages(Constants.TrainImages, Constants.TrainLabels);
        TwoLayerNet network = new TwoLayerNet(784, 50, 10);
        // 正規化されたイメージとone-hotラベルの先頭3個をそれぞれ取り出します。
        INDArray x_batch = train.normalizedImages().get(NDArrayIndex.interval(0, 3));
        INDArray t_batch = train.oneHotLabels().get(NDArrayIndex.interval(0, 3));
        // 勾配を数値微分によって求めます。
        TwoLayerParams grad_numerical = network.numerical_gradient(x_batch, t_batch);
        // 勾配を誤差伝播法によって求めます。
        TwoLayerParams grad_backprop = network.gradient(x_batch, t_batch);
        // 各重みの絶対誤差の平均を求めます。
        System.out.println(grad_numerical);
        System.out.println(grad_backprop);
        double diff_b1 = Functions.average(Transforms.abs(grad_backprop.b1.sub(grad_numerical.b1)));
        double diff_W2 = Functions.average(Transforms.abs(grad_backprop.W2.sub(grad_numerical.W2)));
        double diff_b2 = Functions.average(Transforms.abs(grad_backprop.b2.sub(grad_numerical.b2)));
        double diff_W1 = Functions.average(Transforms.abs(grad_backprop.W1.sub(grad_numerical.W1)));
        System.out.println("b1=" + diff_b1);
        System.out.println("W2=" + diff_W2);
        System.out.println("b2=" + diff_b2);
        System.out.println("W1=" + diff_W1);
        // 最初の結果
        // b1=0.0053852123022079465
        // W2=0.01117962646484375
        // b2=0.27975807189941404
        // W1=9.187315921394192E-4
        assertTrue(diff_b1 < 1e-9);
        assertTrue(diff_W2 < 1e-9);
        assertTrue(diff_b2 < 1e-9);
        assertTrue(diff_W1 < 1e-9);
    }


}
