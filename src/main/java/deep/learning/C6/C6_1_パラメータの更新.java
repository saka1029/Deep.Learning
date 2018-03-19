package deep.learning.C6;

import static org.junit.Assert.*;

import java.awt.Color;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BinaryOperator;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import deep.learning.common.Constants;
import deep.learning.common.GraphImage;
import deep.learning.common.MNISTImages;
import deep.learning.common.Optimizer;
import deep.learning.common.Params;

public class C6_1_パラメータの更新 {

    @Test
    public void C6_1_2_SGD() {
        // Optimizerインタフェースを追加しました。

        // 疑似コードです。
        // TwoLayerNet network = new TwoLayerNet(...);
        // Optimizer optimizer = new SGD();
        // for (int i = 0; i < 10000; ++i) {
        // ...
        // INDArray x_batch = ...
        // INDArray t_batch = ...
        // TwoLayerParams grads = network.gradient(x_batch, t_batch);
        // TwoLayerParams params = network.params;
        // optimizer.update(params, grads);
        // }
    }

    @Test
    public void C6_1_7_どの更新手法を用いるか() throws IOException {
        // ch06/optimizer_compare_naive.py の java版です。
        // GraphImageを使ってグラフを作成します。
        File outdir = Constants.OptimizerImages;
        if (!outdir.exists()) outdir.mkdirs();
        // BinaryOperator<INDArray> f = (x, y) ->
        // x.mul(x).div(y.mul(y).add(20.0));
        BinaryOperator<INDArray> df = (x, y) -> Nd4j.concat(1, x.div(10.0), y.mul(2.0));

        double[] init_pos = new double[] {-7.0, 2.0};
        // 初期値の(0, 0)からの距離です。
        double init_distance = Math.hypot(init_pos[0], init_pos[1]);
        Params params = new Params()
            .put("x", Nd4j.create(new double[] {init_pos[0]}))
            .put("y", Nd4j.create(new double[] {init_pos[1]}));
        Params grads = new Params()
            .put("x", Nd4j.create(new double[] {0}))
            .put("y", Nd4j.create(new double[] {0}));

        Map<String, Optimizer> optimizers = new LinkedHashMap<>();
        optimizers.put("SGD", new SGD(0.95));
        optimizers.put("Momentum", new Momentum(0.1));
        optimizers.put("AdaGrad", new AdaGrad(1.5));
        optimizers.put("Adam", new Adam(0.3));

        for (String key : optimizers.keySet()) {
            Optimizer optimizer = optimizers.get(key);
            params.put("x", Nd4j.create(new double[] {init_pos[0]}))
                .put("y", Nd4j.create(new double[] {init_pos[1]}));
            double min_distance = Double.MAX_VALUE;
            double last_distance = 0.0;
            double prevX = init_pos[0];
            double prevY = init_pos[1];
            try (GraphImage image = new GraphImage(700, 700, -10, -10, 10, 10)) {
                // グラフのタイトルを描画します。
                image.text(key, -2, 7);
                // 最初の点をプロットします。
                image.plot(prevX, prevY);
                for (int i = 0; i < 30; ++i) {
                    INDArray temp = df.apply(params.get("x"), params.get("y"));
                    grads.put("x", temp.getColumn(0));
                    grads.put("y", temp.getColumn(1));
                    optimizer.update(params, grads);
                    double x = params.get("x").getDouble(0);
                    double y = params.get("y").getDouble(0);
                    last_distance = Math.hypot(x, y);
                    if (last_distance < min_distance)
                        min_distance = last_distance;
                    // 直前の点から線を引きます。
                    image.line(prevX, prevY, x, y);
                    // 値をプロットします。
                    image.plot(x, y);
                    prevX = x;
                    prevY = y;
                }
                // 初期値よりも最適化されていることを確認します。
                assertTrue(last_distance < init_distance);
                assertTrue(min_distance < init_distance);
                // グラフをファイル出力します。
                image.writeTo(new File(outdir, key + ".png"));
            }
        }
    }

    @Test
    public void C6_1_8_MNISTデータセットによる更新手法の比較() throws IOException {
        // ch06/optimizer_compare_mnist.py の Java版です。
        MNISTImages train = new MNISTImages(Constants.TrainImages, Constants.TrainLabels);
        INDArray x_train = train.normalizedImages();
        INDArray t_train = train.oneHotLabels();

        int train_size = x_train.size(0);
        int batch_size = 128;
        int max_iterations = 2000;

        // 1.実験の設定
        Map<String, Optimizer> optimizers = new HashMap<>();
        optimizers.put("SGD", new SGD());
        optimizers.put("Momentum", new Momentum());
        optimizers.put("AdaGrad", new AdaGrad());
        optimizers.put("Adam", new Adam());
        // optimizers.put("RMSprop", new RMSprop());

        Map<String, MultiLayerNet> networks = new HashMap<>();
        Map<String, List<Double>> train_loss = new HashMap<>();
        for (String key : optimizers.keySet()) {
            networks.put(key, new MultiLayerNet(
                784, new int[] {100, 100, 100, 100}, 10));
            train_loss.put(key, new ArrayList<>());
        }
        DataSet dataset = new DataSet(x_train, t_train);

        // 2.訓練の開始
        for (int i = 0; i < max_iterations; ++i) {
            // バッチデータを抽出します。
            DataSet sample = dataset.sample(batch_size);
            INDArray x_batch = sample.getFeatureMatrix();
            INDArray t_batch = sample.getLabels();
            for (String key : optimizers.keySet()) {
                MultiLayerNet network = networks.get(key);
                Params grads = network.gradicent(x_batch, t_batch);
                optimizers.get(key).update(network.params, grads);
                double loss = network.loss(x_batch, t_batch);
                train_loss.get(key).add(loss);
            }
            if (i % 100 == 0) {
                System.out.println("===========" + "iteration:" + i + "===========");
                for (String key : optimizers.keySet()) {
                    double loss = networks.get(key).loss(x_batch, t_batch);
                    System.out.println(key + ":" + loss);
                }
            }
        }

        // 3.グラフの描画
        try (GraphImage graph = new GraphImage(1000, 800, -100, -0.1, 2000, 1.0)) {
            Map<String, Color> colors = new HashMap<>();
            colors.put("SGD", Color.GREEN);
            colors.put("Momentum", Color.BLUE);
            colors.put("AdaGrad", Color.RED);
            colors.put("Adam", Color.ORANGE);
            double w = 1300;
            double h = 0.7;
            for (String key : train_loss.keySet()) {
                List<Double> loss = train_loss.get(key);
                graph.color(colors.get(key));
                graph.text(key, w, h);
                h += 0.05;
                graph.plot(0, loss.get(0));
                int step = 10;
                for (int i = step, size = loss.size(); i < size; i += step) {
                    graph.line(i - step, loss.get(i - step), i, loss.get(i));
                    graph.plot(i, loss.get(i));
                }
            }
            graph.color(Color.BLACK);
            graph.text("横=繰り返し回数(0,2000) 縦=損失関数の値(0,1)", w, h);
            h += 0.05;
            graph.text("MNISTデータセットに対する4つの更新手法の比較", w, h);
            if (!Constants.OptimizerImages.exists())
                Constants.OptimizerImages.mkdirs();
            graph.writeTo(new File(Constants.OptimizerImages, "compare_mnist.png"));
        }
    }

}
