package deep.learning.C5;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

import deep.learning.common.Constants;
import deep.learning.common.Functions;
import deep.learning.common.MNISTImages;
import deep.learning.common.Params;
import deep.learning.common.Util;

public class C5_7_誤差逆伝播法の実装 {

    @Test
    public void C5_7_3_誤差逆伝播法の勾配確認() throws Exception {
        // MNISTの訓練データを読み込みます。
        MNISTImages train = new MNISTImages(Constants.TrainImages, Constants.TrainLabels);
        TwoLayerNet network = new TwoLayerNet(784, 50, 10);
        // 正規化されたイメージとone-hotラベルの先頭3個をそれぞれ取り出します。
        int batch_size = 3;
        INDArray x_batch = train.normalizedImages().get(NDArrayIndex.interval(0, batch_size));
        INDArray t_batch = train.oneHotLabels().get(NDArrayIndex.interval(0, batch_size));
        // 勾配を数値微分によって求めます。
        Params grad_numerical = network.numerical_gradient(x_batch, t_batch);
        // 勾配を誤差伝播法によって求めます。
        Params grad_backprop = network.gradient(x_batch, t_batch);
        // 数値微分と誤差伝播法の結果を比較します。
        double diff_W1 = Functions.average(Transforms.abs(grad_backprop.get("W1").sub(grad_numerical.get("W1"))));
        double diff_b1 = Functions.average(Transforms.abs(grad_backprop.get("b1").sub(grad_numerical.get("b1"))));
        double diff_W2 = Functions.average(Transforms.abs(grad_backprop.get("W2").sub(grad_numerical.get("W2"))));
        double diff_b2 = Functions.average(Transforms.abs(grad_backprop.get("b2").sub(grad_numerical.get("b2"))));
        System.out.println("W1=" + diff_W1);
        System.out.println("b1=" + diff_b1);
        System.out.println("W2=" + diff_W2);
        System.out.println("b2=" + diff_b2);
        // 差分は本書より少し大きめです。
        assertTrue(diff_b1 < 1e-3);
        assertTrue(diff_W2 < 1e-3);
        assertTrue(diff_b2 < 1e-3);
        assertTrue(diff_W1 < 1e-3);
    }

    @Test
    public void C5_7_4_誤差逆伝播法を使った学習() throws Exception {
        // MNISTの訓練データを読み込みます。
        MNISTImages train = new MNISTImages(Constants.TrainImages, Constants.TrainLabels);
        INDArray x_train = train.normalizedImages();
        INDArray t_train = train.oneHotLabels();
        // MNISTのテストデータを読み込みます。
        MNISTImages test = new MNISTImages(Constants.TestImages, Constants.TestLabels);
        INDArray x_test = test.normalizedImages();
        INDArray t_test = test.oneHotLabels();
        TwoLayerNet network = new TwoLayerNet(784, 50, 10);
        DataSet dataSet = new DataSet(x_train, t_train);
        int iters_num = 10000;
        int train_size = x_train.size(0);
        int batch_size = 100;
        double learning_rate = 0.1;
        List<Double> train_loss_list = new ArrayList<>();
        List<Double> train_acc_list = new ArrayList<>();
        List<Double> test_acc_list = new ArrayList<>();
        int iter_per_epoch = Math.max(train_size / batch_size, 1);
        for (int i = 0; i < iters_num; ++i) {
            DataSet sample = dataSet.sample(batch_size);
            INDArray x_batch = sample.getFeatures();
            INDArray t_batch = sample.getLabels();
            // 誤差逆伝播法によって勾配を求める
            Params grad = network.gradient(x_batch, t_batch);
            // 更新
            network.params.update((p, a) -> p.subi(a.mul(learning_rate)), grad);
            double loss = network.loss(x_batch, t_batch);
            train_loss_list.add(loss);
            if (i % iter_per_epoch == 0) {
                double train_acc = network.accuracy(x_train, t_train);
                double test_acc = network.accuracy(x_test, t_test);
                train_acc_list.add(train_acc);
                test_acc_list.add(test_acc);
                System.out.printf("loss=%f train_acc=%f test_acc=%f%n", loss, train_acc, test_acc);
            }
        }
        assertTrue(train_acc_list.get(train_acc_list.size() - 1) > 0.8);
        assertTrue(test_acc_list.get(test_acc_list.size() - 1) > 0.8);
    }

    @Test
    public void testSum() {
        INDArray a = Nd4j.create(new double[][] {{1, 2}, {3, 4}});
        // 行を集約
        assertEquals("[4.00,6.00]", Util.string(a.sum(0)));
        // 列を集約
        assertEquals("[3.00,7.00]", Util.string(a.sum(1)));
    }

}
