package deep.learning.C6;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import deep.learning.common.Affine;
import deep.learning.common.Functions;
import deep.learning.common.INDArrayFunction;
import deep.learning.common.LastLayer;
import deep.learning.common.Layer;
import deep.learning.common.Params;
import deep.learning.common.Relu;
import deep.learning.common.Sigmoid;
import deep.learning.common.SoftmaxWithLoss;

/**
 * 全結合による多層ニューラルネットワーク
 */
public class MultiLayerNet {

    final int input_size, output_size, hidden_layer_num;
    final int[] hidden_size_list;
    final double weight_decay_lambda;
    final Params params;
    final Map<String, Layer> layers;
    final LastLayer last_layer;

    /**
     * コンストラクタです。
     *
     * @param input_size 入力サイズ(MNISTの場合は784)
     * @param hidden_size_list 隠れ層のニューロン数の配列(ex. new int[] {100, 100, 100})
     * @param output_size 出力サイズ(MNISTの場合は10)
     * @param activation "relu" or "sigmoid"
     * @param weight_init_std 重みの標準偏差を指定(e.g. 0.01)
     *                        "relu"または"he"を指定した場合は「Heの初期値」を設定
     *                        "sigmoid"または"xavier"を指定した場合は「Xavierの初期値」を設定
     * @param weight_decay_lambda Weight Decay (L2ノルム)の強さ
     */
    public MultiLayerNet(int input_size, int[] hidden_size_list, int output_size,
        String activation, double weight_init_std, double weight_decay_lambda) {
        this.input_size = input_size;
        this.output_size = output_size;
        this.hidden_size_list = hidden_size_list;
        this.hidden_layer_num = hidden_size_list.length;
        this.weight_decay_lambda = weight_decay_lambda;
        this.params = new Params();
        // 重みの初期化
        __init_weight(activation, weight_init_std);
        // レイヤの生成
        Map<String, Supplier<Layer>> activation_layer = new LinkedHashMap<>();
        activation_layer.put("sigmoid", Sigmoid::new);
        activation_layer.put("relu", Relu::new);
        layers = new LinkedHashMap<>();
        for (int idx = 1; idx < hidden_layer_num + 1; ++idx) {
            layers.put("Affine" + idx, new Affine(params.get("W" + idx),
                                                  params.get("b" + idx)));
            layers.put("Activation_function" + idx, activation_layer.get(activation).get());
        }
        int idx = hidden_layer_num + 1;
        layers.put("Affine" + idx, new Affine(params.get("W" + idx),
                                              params.get("b" + idx)));
        last_layer = new SoftmaxWithLoss();
    }

    public MultiLayerNet(int input_size, int[] hidden_size_list, int output_size) {
        this(input_size, hidden_size_list, output_size, "relu", 0.01, 0);
    }

    /**
     * 重みの初期値設定
     * @param activation
     * @param weight_init_std 重みの標準偏差を指定(e.g. 0.01)
     *                        "relu"または"he"を指定した場合は「Heの初期値」を設定
     *                        "sigmoid"または"xavier"を指定した場合は「Xavierの初期値」を設定
     */
    private void __init_weight(String activation, double weight_init_std) {
        int[] all_size_list = new int[hidden_size_list.length + 2];
        all_size_list[0] = input_size;
        System.arraycopy(hidden_size_list, 0, all_size_list, 1, hidden_size_list.length);
        all_size_list[all_size_list.length - 1] = output_size;
        for (int idx = 1; idx < all_size_list.length; ++idx) {
            double scale = weight_init_std;
            switch (activation.toLowerCase()) {
            case "relu": case "he":
                scale = Math.sqrt(2.0 / all_size_list[idx - 1]); // ReLUを使う場合に推奨される初期値
                break;
            case "sigmoid": case "xavier":
                scale = Math.sqrt(1.0 / all_size_list[idx - 1]); // sigmoidを使う場合に推奨される初期値
                break;
            }
            params.put("W" + idx, Nd4j.randn(all_size_list[idx -1], all_size_list[idx]).mul(scale));
            params.put("b" + idx, Nd4j.zeros(all_size_list[idx]));
        }

    }

    public INDArray predict(INDArray x) {
        for (Layer layer : layers.values())
            x = layer.forward(x);
        return x;
    }

    /**
     * 損失関数を求める。
     *
     * @param x 入力データ
     * @param t 教師ラベル
     * @return 損失関数の値
     */
    public double loss(INDArray x, INDArray t) {
        INDArray y = predict(x);
        double weight_decay = 0;
        for (int idx = 1; idx < hidden_layer_num + 2; ++idx) {
            INDArray W = params.get("W" + idx);
            weight_decay += 0.5 * weight_decay_lambda * W.mul(W).sumNumber().doubleValue();
        }
        return last_layer.forward(y, t) + weight_decay;
    }

    public double accuracy(INDArray x, INDArray t) {
        INDArray y = predict(x);
        y = Functions.argmax(y);
        if (t.size(0) != 1)
            t = Functions.argmax(t);
        double accuracy = y.eq(t).sumNumber().doubleValue() / x.size(0);
        return accuracy;
    }

    /**
     * 勾配を求める（数値微分）
     *
     * @param x 入力データ
     * @param t 教師ラベル
     * @return 各層の勾配を持ったParams
     *         grads.get("W1"), grads.get("W1"), ... は各層の重み
     *         grads.get("b1"), grads.get("b1"), ... は各層のバイアス
     */
    public Params numerical_gradient(INDArray x, INDArray t) {
        INDArrayFunction loss_W = W -> loss(x, t);
        Params grads = new Params();
        for (int idx = 1; idx < hidden_layer_num + 2; ++idx) {
            grads.put("W" + idx, Functions.numerical_gradient(loss_W, params.get("W" + idx)));
            grads.put("b" + idx, Functions.numerical_gradient(loss_W, params.get("b" + idx)));
        }
        return grads;
    }

    /**
     * 勾配を求める（誤差逆伝播法）
     *
     * @param x 入力データ
     * @param t 教師ラベル
     * @return 各層の勾配を持ったParams
     *         grads.get("W1"), grads.get("W1"), ... は各層の重み
     *         grads.get("b1"), grads.get("b1"), ... は各層のバイアス
     */
    public Params gradicent(INDArray x, INDArray t) {
        // forward
        loss(x, t);
        // backward
        INDArray dout = Nd4j.create(new double[] {1});
        // last_layerはdoutの値を参照しない点に注意してください。
        dout = last_layer.backward(dout);
        List<Layer> layers = new ArrayList<>(this.layers.values());
        Collections.reverse(layers);
        for (Layer layer : layers)
            dout = layer.backward(dout);
        Params grads = new Params();
        for (int idx = 1; idx < hidden_layer_num + 2; ++idx) {
            grads.put("W" + idx, ((Affine)this.layers.get("Affine" + idx)).dW
                .add(((Affine)this.layers.get("Affine" + idx)).W.mul(weight_decay_lambda)));
            grads.put("b" + idx, ((Affine)this.layers.get("Affine" + idx)).db);
        }
        return grads;
    }
}
