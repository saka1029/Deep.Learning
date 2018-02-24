package deep.learning.C4;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.List;

import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.dataset.DataSet;

import deep.learning.common.Constants;
import deep.learning.common.MNISTImages;
import deep.learning.common.TwoLayerNet;
import deep.learning.common.TwoLayerParams;

public class C4_5_学習アルゴリズムの実装 {

    @Test
    @Ignore // ７分くらいかかります。
    public void C4_5_1_２層ニューラルネットワークのクラス() throws Exception {
        // deep.learning.common.TwoLayerNet.java参照
        TwoLayerNet net = new TwoLayerNet(784, 100, 10);
        assertArrayEquals(new int[] {784, 100}, net.parms.W1.shape());
        assertArrayEquals(new int[] {1, 100}, net.parms.b1.shape());
        assertArrayEquals(new int[] {100, 10}, net.parms.W2.shape());
        assertArrayEquals(new int[] {1, 10}, net.parms.b2.shape());
        try (Random r = new DefaultRandom()) {
            INDArray x = r.nextGaussian(new int[] {100, 784});
            INDArray t = r.nextGaussian(new int[] {100, 10});
            INDArray y = net.predict(x);
            assertArrayEquals(new int[] {1, 10}, y.shape());
            TwoLayerParams grads = net.numerical_gradient(x, t);
            assertArrayEquals(new int[] {784, 100}, grads.W1.shape());
            assertArrayEquals(new int[] {1, 100}, grads.b1.shape());
            assertArrayEquals(new int[] {100, 10}, grads.W2.shape());
            assertArrayEquals(new int[] {1, 10}, grads.b2.shape());
        }
    }

    /**
     * 本書のサンプル通り実行すると非常に時間がかかります。
     * batch_size = 10で実行すると
     * 以下のように損失があまり収束しません。
     *
     * iteration 0 loss=21.161320 elapse=39989ms
     * iteration 1 loss=24.797092 elapse=39026ms
     * iteration 2 loss=15.007333 elapse=41412ms
     * iteration 3 loss=21.884285 elapse=41177ms
     * iteration 4 loss=22.502634 elapse=36414ms
     * iteration 5 loss=21.690832 elapse=36761ms
     * iteration 6 loss=20.682486 elapse=37022ms
     * iteration 7 loss=22.309616 elapse=36583ms
     * iteration 8 loss=19.859619 elapse=36509ms
     * iteration 9 loss=21.543022 elapse=36410ms
     * iteration 10 loss=21.059685 elapse=35880ms
     * iteration 11 loss=21.272987 elapse=35876ms
     * iteration 12 loss=21.703112 elapse=35702ms
     * iteration 13 loss=21.226400 elapse=36229ms
     * iteration 14 loss=22.242090 elapse=37570ms
     * iteration 15 loss=20.497494 elapse=36254ms
     * iteration 16 loss=21.165102 elapse=36615ms
     * iteration 17 loss=22.429100 elapse=36116ms
     * iteration 18 loss=20.541040 elapse=36535ms
     * iteration 19 loss=21.336864 elapse=36361ms
     * iteration 20 loss=19.254723 elapse=36316ms
     * iteration 21 loss=19.810532 elapse=36208ms
     * iteration 22 loss=20.649469 elapse=36045ms
     * iteration 23 loss=18.904522 elapse=36584ms
     * iteration 24 loss=21.976421 elapse=36079ms
     * iteration 25 loss=20.250872 elapse=35932ms
     * iteration 26 loss=20.143461 elapse=36069ms
     * iteration 27 loss=20.679714 elapse=36178ms
     * iteration 28 loss=22.703018 elapse=36371ms
     * iteration 29 loss=20.430645 elapse=36210ms
     * iteration 30 loss=19.426363 elapse=37021ms
     * iteration 31 loss=19.872597 elapse=35873ms
     * iteration 32 loss=18.279284 elapse=37494ms
     * iteration 33 loss=17.846779 elapse=37029ms
     * iteration 34 loss=22.402769 elapse=37663ms
     * iteration 35 loss=18.610935 elapse=37216ms
     * iteration 36 loss=19.373077 elapse=38129ms
     * iteration 37 loss=21.684448 elapse=39237ms
     * iteration 38 loss=13.853874 elapse=37740ms
     * iteration 39 loss=20.154537 elapse=38260ms
     * iteration 40 loss=19.698437 elapse=38143ms
     * iteration 41 loss=16.229975 elapse=41870ms
     * iteration 42 loss=19.412991 elapse=35885ms
     * iteration 43 loss=17.263290 elapse=37913ms
     * iteration 44 loss=20.922745 elapse=37139ms
     * @throws Exception
     */
    @Ignore // batch_sizeを10としてもfor文1回あたり30秒以上かかる。
    @Test
    public void C4_5_2_ミニバッチ学習の実装() throws Exception {
        // MNISTデータセットを読み込みます。
        MNISTImages train = new MNISTImages(Constants.TrainImages, Constants.TrainLabels);
        INDArray images = train.normalizedImages();
        INDArray labels = train.oneHotLabels();
        assertArrayEquals(new int[] {60000, 784}, images.shape());
        assertArrayEquals(new int[] {60000, 10}, labels.shape());
        List<Double> train_loss_list =  new ArrayList<>();
        int iters_num = 10000;
        int train_size = images.size(0);
        int batch_size = 10;
//        int batch_size = 1000;
        double learning_rate = 0.1;
        TwoLayerNet network = new TwoLayerNet(784, 50, 10);
        // batch_size分のデータをランダムに取り出します。
        for (int i = 0; i < iters_num; ++i) {
            long start = System.currentTimeMillis();
            // ミニバッチの取得
            DataSet ds = new DataSet(train.normalizedImages(), train.oneHotLabels());
            DataSet sample = ds.sample(batch_size);
            INDArray x_batch = sample.getFeatureMatrix();
            INDArray t_batch = sample.getLabels();
            TwoLayerParams grad =  network.numerical_gradient(x_batch, t_batch);
            network.parms.W1.subi(grad.W1.mul(learning_rate));
            network.parms.b1.subi(grad.b1.mul(learning_rate));
            network.parms.W2.subi(grad.W2.mul(learning_rate));
            network.parms.b2.subi(grad.b2.mul(learning_rate));
            // 学習経過の記録
            double loss = network.loss(x_batch, t_batch);
            train_loss_list.add(loss);
            System.out.printf("iteration %d loss=%f elapse=%dms%n",
                i, loss, System.currentTimeMillis() - start);
        }
    }
}
