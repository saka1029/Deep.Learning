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
    @Ignore // 5～10分くらいかかります。
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
            assertArrayEquals(new int[] {100, 10}, y.shape());
            TwoLayerParams grads = net.numerical_gradient(x, t);
            assertArrayEquals(new int[] {784, 100}, grads.W1.shape());
            assertArrayEquals(new int[] {1, 100}, grads.b1.shape());
            assertArrayEquals(new int[] {100, 10}, grads.W2.shape());
            assertArrayEquals(new int[] {1, 10}, grads.b2.shape());
        }
    }

    /**
     * 本書のサンプル通り実行すると非常に時間がかかります。
     */
    @Ignore // ループ1回につき90秒程度10000回ループすると10日程度になる見込みです。
    @Test
    public void C4_5_2_ミニバッチ学習の実装() throws Exception {
        // MNISTデータセットを読み込みます。
        MNISTImages train = new MNISTImages(Constants.TrainImages, Constants.TrainLabels);
        INDArray x_train = train.normalizedImages();
        INDArray t_train = train.oneHotLabels();
        assertArrayEquals(new int[] {60000, 784}, x_train.shape());
        assertArrayEquals(new int[] {60000, 10}, t_train.shape());
        List<Double> train_loss_list =  new ArrayList<>();
        int iters_num = 10000;
        // int train_size = images.size(0);
         int batch_size = 100;
        double learning_rate = 0.1;
        TwoLayerNet network = new TwoLayerNet(784, 50, 10);
        // batch_size分のデータをランダムに取り出します。
        for (int i = 0; i < iters_num; ++i) {
            long start = System.currentTimeMillis();
            // ミニバッチの取得
            DataSet ds = new DataSet(x_train, t_train);
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

    @Ignore
    @Test
    public void C4_5_3_テストデータで評価() throws Exception {
        // MNISTデータセットを読み込みます。
        MNISTImages train = new MNISTImages(Constants.TrainImages, Constants.TrainLabels);
        INDArray x_train = train.normalizedImages();
        INDArray t_train = train.oneHotLabels();
        MNISTImages test = new MNISTImages(Constants.TestImages, Constants.TestLabels);
        INDArray x_test = test.normalizedImages();
        INDArray t_test = test.oneHotLabels();
        assertArrayEquals(new int[] {60000, 784}, x_train.shape());
        assertArrayEquals(new int[] {60000, 10}, t_train.shape());
        List<Double> train_loss_list =  new ArrayList<>();
        List<Double> train_acc_list = new ArrayList<>();
        List<Double> test_acc_list = new ArrayList<>();
        int iters_num = 10000;
        int train_size = x_train.size(0);
//        int batch_size = 50;
         int batch_size = 100;
        double learning_rate = 0.01;
        int iter_per_epoch = Math.max(train_size / batch_size, 1);
        TwoLayerNet network = new TwoLayerNet(784, 50, 10);
        // batch_size分のデータをランダムに取り出します。
        for (int i = 0; i < iters_num; ++i) {
            long start = System.currentTimeMillis();
            // ミニバッチの取得
            DataSet ds = new DataSet(x_train, t_train);
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
            // 1エポックごとに認識制度を計算
//            if (i % iter_per_epoch == 0) {
                double train_acc = network.accuracy(x_train, t_train);
                double test_acc = network.accuracy(x_test, t_test);
                train_acc_list.add(train_acc);
                test_acc_list.add(test_acc);
                System.out.printf("train acc, test acc | %s, %s%n",
                    train_acc, test_acc);
//            }
            System.out.printf("iteration %d loss=%f elapse=%dms%n",
                i, loss, System.currentTimeMillis() - start);
            // 実行例
            // train acc, test acc | 0.13663333333333333, 0.1377
            // iteration 0 loss=227.292389 elapse=130824ms
            // train acc, test acc | 0.0993, 0.1032
            // iteration 1 loss=224.047760 elapse=119176ms
            // train acc, test acc | 0.09915, 0.1009
            // iteration 2 loss=230.238113 elapse=106287ms
            // train acc, test acc | 0.11236666666666667, 0.1135
            // iteration 3 loss=221.465485 elapse=103833ms
            // train acc, test acc | 0.09736666666666667, 0.0982
            // iteration 4 loss=228.571060 elapse=100573ms
            // train acc, test acc | 0.09736666666666667, 0.0982
            // iteration 5 loss=224.213898 elapse=103367ms
            // train acc, test acc | 0.09871666666666666, 0.098
            // iteration 6 loss=224.962784 elapse=103401ms
            // train acc, test acc | 0.09915, 0.1009
            // iteration 7 loss=226.428650 elapse=102678ms
            // train acc, test acc | 0.11236666666666667, 0.1135
            // iteration 8 loss=221.897980 elapse=105195ms
            // train acc, test acc | 0.10516666666666667, 0.1039
            // iteration 9 loss=223.947723 elapse=105490ms
        }
    }
}
