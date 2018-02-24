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
//    @Ignore // ７分くらいかかります。
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
            // train acc, test acc | 0.10441666666666667, 0.1028
            // iteration 10 loss=222.308884 elapse=102915ms
            // train acc, test acc | 0.26048333333333334, 0.2591
            // iteration 11 loss=223.157288 elapse=109817ms
            // train acc, test acc | 0.09915, 0.1009
            // iteration 12 loss=224.265045 elapse=112302ms
            // train acc, test acc | 0.14118333333333333, 0.148
            // iteration 13 loss=222.802933 elapse=109715ms
            // train acc, test acc | 0.14821666666666666, 0.1507
            // iteration 14 loss=224.678253 elapse=111178ms
            // train acc, test acc | 0.17195, 0.1747
            // iteration 15 loss=223.122009 elapse=107495ms
            // train acc, test acc | 0.2568, 0.2633
            // iteration 16 loss=224.869354 elapse=104954ms
            // train acc, test acc | 0.2688333333333333, 0.2722
            // iteration 17 loss=222.433716 elapse=105454ms
            // train acc, test acc | 0.22555, 0.2263
            // iteration 18 loss=218.940552 elapse=3186789ms
            // train acc, test acc | 0.2586333333333333, 0.2704
            // iteration 19 loss=216.079895 elapse=85122ms
            // train acc, test acc | 0.21981666666666666, 0.2273
            // iteration 20 loss=214.871933 elapse=82135ms
            // train acc, test acc | 0.11266666666666666, 0.1142
            // iteration 21 loss=207.320160 elapse=81635ms
            // train acc, test acc | 0.27586666666666665, 0.2904
            // iteration 22 loss=209.097580 elapse=79995ms
            // train acc, test acc | 0.3412833333333333, 0.3426
            // iteration 23 loss=213.892242 elapse=80009ms
            // train acc, test acc | 0.29313333333333336, 0.2952
            // iteration 24 loss=205.810501 elapse=80165ms
            // train acc, test acc | 0.21305, 0.2167
            // iteration 25 loss=198.719543 elapse=80266ms
            // train acc, test acc | 0.2693, 0.2754
            // iteration 26 loss=198.343964 elapse=80260ms
            // train acc, test acc | 0.3089166666666667, 0.3187
            // iteration 27 loss=199.445984 elapse=79791ms
            // train acc, test acc | 0.4273666666666667, 0.4364
            // iteration 28 loss=200.976059 elapse=79900ms
            // train acc, test acc | 0.4782666666666667, 0.4754
            // iteration 29 loss=186.964203 elapse=79756ms
            // train acc, test acc | 0.30775, 0.3117
            // iteration 30 loss=178.232224 elapse=80009ms
            // train acc, test acc | 0.35108333333333336, 0.3525
            // iteration 31 loss=172.872864 elapse=79897ms
            // train acc, test acc | 0.3181333333333333, 0.3145
            // iteration 32 loss=171.593475 elapse=79400ms
            // train acc, test acc | 0.33636666666666665, 0.3348
            // iteration 33 loss=170.547440 elapse=79621ms
            // train acc, test acc | 0.38008333333333333, 0.3826
            // iteration 34 loss=170.023056 elapse=79866ms
            // train acc, test acc | 0.44825, 0.4485
            // iteration 35 loss=156.487946 elapse=80088ms
            // train acc, test acc | 0.5011166666666667, 0.5022
            // iteration 36 loss=157.098160 elapse=79882ms
            // train acc, test acc | 0.45798333333333335, 0.455
            // iteration 37 loss=149.828888 elapse=79900ms
            // train acc, test acc | 0.47281666666666666, 0.4711
            // iteration 38 loss=139.964691 elapse=79510ms
            // train acc, test acc | 0.6249, 0.6393
            // iteration 39 loss=152.069275 elapse=79830ms
            // train acc, test acc | 0.5553333333333333, 0.5622
            // iteration 40 loss=140.551865 elapse=79287ms
            // train acc, test acc | 0.5327, 0.5421
            // iteration 41 loss=143.895065 elapse=80275ms
            // train acc, test acc | 0.5368166666666667, 0.5558
            // iteration 42 loss=135.750565 elapse=79947ms
            // train acc, test acc | 0.6562, 0.6675
            // iteration 43 loss=136.225479 elapse=79475ms
            // train acc, test acc | 0.6099, 0.619
            // iteration 44 loss=127.619629 elapse=79447ms
            // train acc, test acc | 0.4678833333333333, 0.4768
            // iteration 45 loss=127.794548 elapse=79452ms
            // train acc, test acc | 0.5706, 0.5763
            // iteration 46 loss=128.328613 elapse=80619ms
            // train acc, test acc | 0.6029, 0.6059
            // iteration 47 loss=122.006592 elapse=80319ms
            // train acc, test acc | 0.5514333333333333, 0.5593
            // iteration 48 loss=106.420700 elapse=80416ms
            // train acc, test acc | 0.5608, 0.5596
            // iteration 49 loss=104.738770 elapse=80897ms
            // train acc, test acc | 0.62805, 0.6306
            // iteration 50 loss=102.543701 elapse=80853ms
            // train acc, test acc | 0.6913666666666667, 0.7013
            // iteration 51 loss=112.201416 elapse=80322ms
            // train acc, test acc | 0.6426666666666667, 0.6502
            // iteration 52 loss=118.906876 elapse=80341ms
            // train acc, test acc | 0.5898, 0.5939
            // iteration 53 loss=109.649666 elapse=80791ms
            // train acc, test acc | 0.6547833333333334, 0.6532
            // iteration 54 loss=107.395866 elapse=79943ms
            // train acc, test acc | 0.6802333333333334, 0.6829
            // iteration 55 loss=101.310226 elapse=80232ms
            // train acc, test acc | 0.7477, 0.7553
            // iteration 56 loss=95.166000 elapse=79555ms
            // train acc, test acc | 0.6757166666666666, 0.6818
            // iteration 57 loss=96.526619 elapse=79432ms
            // train acc, test acc | 0.6694166666666667, 0.6734
            // iteration 58 loss=111.847763 elapse=79603ms
            // train acc, test acc | 0.67955, 0.6889
            // iteration 59 loss=86.365601 elapse=79665ms
            // train acc, test acc | 0.7017333333333333, 0.7074
            // iteration 60 loss=91.731216 elapse=79697ms
            // train acc, test acc | 0.7403333333333333, 0.7477
            // iteration 61 loss=90.900162 elapse=79541ms
            // train acc, test acc | 0.6494666666666666, 0.6524
            // iteration 62 loss=88.385178 elapse=80082ms
            // train acc, test acc | 0.6679833333333334, 0.6728
            // iteration 63 loss=79.298721 elapse=79725ms
            // train acc, test acc | 0.7378833333333333, 0.7514
            // iteration 64 loss=91.141266 elapse=80322ms
            // train acc, test acc | 0.6974833333333333, 0.7024
            // iteration 65 loss=85.601227 elapse=82401ms
            // train acc, test acc | 0.6924333333333333, 0.7012
            // iteration 66 loss=78.345589 elapse=97715ms
            // train acc, test acc | 0.7726833333333334, 0.7791
            // iteration 67 loss=84.396355 elapse=99093ms
            // train acc, test acc | 0.7399166666666667, 0.7472
            // iteration 68 loss=79.798416 elapse=97939ms
            // train acc, test acc | 0.7522333333333333, 0.7597
            // iteration 69 loss=80.401886 elapse=101151ms
            // train acc, test acc | 0.7855666666666666, 0.7903
            // iteration 70 loss=83.157684 elapse=101856ms
            // train acc, test acc | 0.7608, 0.7639
            // iteration 71 loss=74.890015 elapse=100297ms
            // train acc, test acc | 0.761, 0.766
            // iteration 72 loss=82.796043 elapse=96701ms
            // train acc, test acc | 0.7675333333333333, 0.7716
            // iteration 73 loss=71.480896 elapse=94394ms
            // train acc, test acc | 0.7953, 0.7984
            // iteration 74 loss=86.362747 elapse=95266ms
            // train acc, test acc | 0.8056166666666666, 0.8126
            // iteration 75 loss=75.059753 elapse=93620ms
        }
    }
}
