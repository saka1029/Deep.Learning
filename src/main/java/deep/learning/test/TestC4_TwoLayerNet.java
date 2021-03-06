package deep.learning.test;

import static org.junit.Assert.*;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;

import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import deep.learning.C4.TwoLayerNet;
import deep.learning.common.Constants;
import deep.learning.common.Functions;
import deep.learning.common.MNISTImages;
import deep.learning.common.Params;

public class TestC4_TwoLayerNet {

    @Test
    public void testTwoLayerNet() throws Exception {
        TwoLayerNet net = new deep.learning.C4.TwoLayerNet(784, 100, 10);
        assertArrayEquals(new int[] {784, 100}, net.parms.get("W1").shape());
        assertArrayEquals(new int[] {1, 100}, net.parms.get("b1").shape());
        assertArrayEquals(new int[] {100, 10}, net.parms.get("W2").shape());
        assertArrayEquals(new int[] {1, 10}, net.parms.get("b2").shape());
    }

    @Test
    public void testAccuracy() throws Exception {
        TwoLayerNet net = new TwoLayerNet(
            Nd4j.create(new double[][] {{1, 1}, {1, 1}}),
            Nd4j.create(new double[] {1, 1}),
            Nd4j.create(new double[][] {{1, 1}, {1, 1}}),
            Nd4j.create(new double[] {1, 1}));
        INDArray x = Nd4j.create(new double[][] {{1, 0}, {0, 1}});
        INDArray t = Nd4j.create(new double[][] {{1, 0}, {0, 1}});
        double accuracy = net.accuracy(x, t);
        assertEquals(0.5, accuracy, 5e-6);
    }

    /**
     * 認識精度が0.9以上になるまで学習を繰り返します。
     * 学習が終了した時点でウェイトをファイルに出力します。
     */
    @Test
    @Ignore //  4～5時間かかります。
    public void testLearn() throws Exception {
        double accuracy_goal = 0.9;
        // ウェイトの出力ディレクトリを確保します。
        if (!Constants.WEIGHTS.exists())
            Constants.WEIGHTS.mkdirs();
        // MNIST訓練データセットを読み込みます。
        MNISTImages train = new MNISTImages(Constants.TrainImages, Constants.TrainLabels);
        INDArray x_train = train.normalizedImages();
        INDArray t_train = train.oneHotLabels();
        // MNISTテストデータセットを読み込みます。
        MNISTImages test = new MNISTImages(Constants.TestImages, Constants.TestLabels);
        INDArray x_test = test.normalizedImages();
        INDArray t_test = test.oneHotLabels();
        assertArrayEquals(new int[] {60000, 784}, x_train.shape());
        assertArrayEquals(new int[] {60000, 10}, t_train.shape());
        List<Double> train_loss_list = new ArrayList<>();
        List<Double> train_acc_list = new ArrayList<>();
        List<Double> test_acc_list = new ArrayList<>();
        int iters_num = 10000;
        int train_size = x_train.size(0);
        int batch_size = 100;
        double learning_rate = 0.01;
        TwoLayerNet network = new TwoLayerNet(784, 50, 10);
        for (int i = 0; i < iters_num; ++i) {
            long start = System.currentTimeMillis();
            // ミニバッチの取得
            DataSet ds = new DataSet(x_train, t_train);
            DataSet sample = ds.sample(batch_size);
            INDArray x_batch = sample.getFeatureMatrix();
            INDArray t_batch = sample.getLabels();
            Params grad = network.numerical_gradient(x_batch, t_batch);
            network.parms.update((p, a) -> p.subi(a.mul(learning_rate)), grad);
            // 学習経過の記録
            double loss = network.loss(x_batch, t_batch);
            double train_acc = network.accuracy(x_train, t_train);
            double test_acc = network.accuracy(x_test, t_test);
            train_acc_list.add(train_acc);
            test_acc_list.add(test_acc);
            System.out.printf("%6d : loss=%f train acc=%.2f test acc=%.2f elapse=%dms%n",
                i, loss, train_acc, test_acc, System.currentTimeMillis() - start);
            if (train_acc >= accuracy_goal && test_acc >= accuracy_goal)
                break;
        }
        // ウェイトの出力ファイルです。
        File weights = new File(Constants.WEIGHTS, "TwoLayerParms.ser");
        // ウェイトをファイルに出力します。
        try (ObjectOutputStream os = new ObjectOutputStream(
            new FileOutputStream(weights))) {
            os.writeObject(network.parms);
        }
    }

    /**
     * 90%まで訓練したウェイトデータで訓練データを認識して
     * 認識精度が90%を超えることを確認します。
     */
    @Ignore
    @Test
    public void testPredict() throws Exception {
        MNISTImages train = new MNISTImages(Constants.TrainImages, Constants.TrainLabels);
        INDArray x_train = train.normalizedImages();
        File weights = new File(Constants.WEIGHTS, "TwoLayerParms.ser");
        Params parms = null;
        try (ObjectInputStream is = new ObjectInputStream(new FileInputStream(weights))) {
            parms = (Params)is.readObject();
        }
        TwoLayerNet network = new TwoLayerNet(parms.get("W1"), parms.get("b1"), parms.get("W2"), parms.get("b2"));
        int batch_size = 100;
        int size = x_train.size(0);
        int accuracy_cnt = 0;
        for (int i = 0; i < size; i += batch_size) {
            // バッチサイズ分のイメージを取り出してpredict()を呼びます。
            INDArray y = network.predict(x_train.get(NDArrayIndex.interval(i, i + batch_size)));
            INDArray max = Functions.argmax(y);
            for (int j = 0; j < batch_size; ++j)
                if (max.getInt(j) == train.label(i + j))
                    ++accuracy_cnt;
        }
        System.out.printf("accuracy = %f (%d/%d)%n", (double)accuracy_cnt / size, accuracy_cnt, size);
        assertEquals(60000, size);
        assertTrue(accuracy_cnt > size * 0.9);
    }
}
