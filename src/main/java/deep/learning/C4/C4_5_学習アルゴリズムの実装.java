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
import deep.learning.common.Params;

public class C4_5_学習アルゴリズムの実装 {

    @Ignore // 5～10分くらいかかります。
    @Test
    public void C4_5_1_２層ニューラルネットワークのクラス() throws Exception {
        // deep.learning.common."java参照
        TwoLayerNet net = new TwoLayerNet(784, 100, 10);
        assertArrayEquals(new int[] {784, 100}, net.parms.get("W1").shape());
        assertArrayEquals(new int[] {1, 100}, net.parms.get("b1").shape());
        assertArrayEquals(new int[] {100, 10}, net.parms.get("W2").shape());
        assertArrayEquals(new int[] {1, 10}, net.parms.get("b2").shape());
        try (Random r = new DefaultRandom()) {
            INDArray x = r.nextGaussian(new int[] {100, 784});
            INDArray t = r.nextGaussian(new int[] {100, 10});
            INDArray y = net.predict(x);
            assertArrayEquals(new int[] {100, 10}, y.shape());
            Params grads = net.numerical_gradient(x, t);
            assertArrayEquals(new int[] {784, 100}, grads.get("W1").shape());
            assertArrayEquals(new int[] {1, 100}, grads.get("b1").shape());
            assertArrayEquals(new int[] {100, 10}, grads.get("W2").shape());
            assertArrayEquals(new int[] {1, 10}, grads.get("b2").shape());
        }
    }

    /**
     * 本書のサンプル通り実行すると非常に時間がかかります。
     */
//    @Ignore // ループ1回につき150秒程度10000回ループすると17日程度になる見込みです。
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
            Params grad =  network.numerical_gradient(x_batch, t_batch);
            network.parms.update((p, a) -> p.subi(a.mul(learning_rate)), grad);
            // 学習経過の記録
            double loss = network.loss(x_batch, t_batch);
            train_loss_list.add(loss);
            System.out.printf("iteration %d loss=%f elapse=%dms%n",
                i, loss, System.currentTimeMillis() - start);
        }
    }

//    @Ignore
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
            Params grad =  network.numerical_gradient(x_batch, t_batch);
            network.parms.update((p, a) -> p.subi(a.mul(learning_rate)), grad);
            // 学習経過の記録
            double loss = network.loss(x_batch, t_batch);
            train_loss_list.add(loss);
            // 1エポックごとに認識制度を計算
            if (i % iter_per_epoch == 0) {
                double train_acc = network.accuracy(x_train, t_train);
                double test_acc = network.accuracy(x_test, t_test);
                train_acc_list.add(train_acc);
                test_acc_list.add(test_acc);
                System.out.printf("train acc, test acc | %s, %s%n",
                    train_acc, test_acc);
            }
            System.out.printf("iteration %d loss=%f elapse=%dms%n",
                i, loss, System.currentTimeMillis() - start);
        }
    }
}
