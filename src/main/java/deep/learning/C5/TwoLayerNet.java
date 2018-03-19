package deep.learning.C5;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.DefaultRandom;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;

import deep.learning.common.Affine;
import deep.learning.common.Functions;
import deep.learning.common.INDArrayFunction;
import deep.learning.common.LastLayer;
import deep.learning.common.Layer;
import deep.learning.common.Params;
import deep.learning.common.Relu;
import deep.learning.common.SoftmaxWithLoss;

public class TwoLayerNet {

    final Params params;
    final Map<String, Layer> layers;
    final LastLayer lastLayer;

    public TwoLayerNet(int input_size, int hidden_size,
        int output_size) throws Exception {
        this(input_size, hidden_size, output_size, 0.01D);
    }

    public TwoLayerNet(int input_size, int hidden_size,
        int output_size, double weight_init_std) throws Exception {
        // 重みの初期化
        // シードをゼロに固定して常に同じ乱数を発生させます。
        try (Random r = new DefaultRandom(0)) {
            params = new Params()
                .put("W1", r.nextGaussian(new int[] {input_size, hidden_size}).mul(weight_init_std))
                .put("b1", Nd4j.zeros(hidden_size))
                .put("W2", r.nextGaussian(new int[] {hidden_size, output_size}).mul(weight_init_std))
                .put("b2", Nd4j.zeros(output_size));
        }
        // レイヤの生成
        layers = new LinkedHashMap<>();
        layers.put("Affine1", new Affine(params.get("W1"), params.get("b1")));
        layers.put("Relu1", new Relu());
        layers.put("Affine2", new Affine(params.get("W2"), params.get("b2")));
        lastLayer = new SoftmaxWithLoss();
    }

    public INDArray predict(INDArray x) {
        for (Layer layer : layers.values())
            x = layer.forward(x);
        return x;
    }

    public double loss(INDArray x, INDArray t) {
        INDArray y = predict(x);
        return lastLayer.forward(y, t);
    }

    public double accuracy(INDArray x, INDArray t) {
        INDArray y = predict(x);
        y = Functions.argmax(y);
        if (t.size(0) != 1)
            t = Functions.argmax(t);
        double accuracy = y.eq(t.broadcast(y.shape())).sumNumber().doubleValue() / x.size(0);
//        double accuracy = y.eps(t.broadcast(y.shape())).sumNumber().doubleValue() / x.size(0);
        return accuracy;
    }

    public Params numerical_gradient(INDArray x, INDArray t) {
        INDArrayFunction loss_W = W -> loss(x, t);
        Params grads = new Params()
            .put("W1", Functions.numerical_gradient(loss_W, params.get("W1")))
            .put("b1", Functions.numerical_gradient(loss_W, params.get("b1")))
            .put("W2", Functions.numerical_gradient(loss_W, params.get("W2")))
            .put("b2", Functions.numerical_gradient(loss_W, params.get("b2")));
        return grads;
    }

    public Params gradient(INDArray x, INDArray t) {
        // forward
        loss(x, t);
        // backward
        INDArray dout = Nd4j.create(new double[] {1});
        // lastLayer:SoftmaxWithLoss.backward(dout)では
        // 引数doutを参照していません。
        dout = lastLayer.backward(dout);
        List<Layer> layers = new ArrayList<>(this.layers.values());
        Collections.reverse(layers);
        for (Layer layer : layers)
            dout = layer.backward(dout);
        Params grads = new Params()
            .put("W1", ((Affine)this.layers.get("Affine1")).dW)
            .put("b1", ((Affine)this.layers.get("Affine1")).db)
            .put("W2", ((Affine)this.layers.get("Affine2")).dW)
            .put("b2", ((Affine)this.layers.get("Affine2")).db);
        return grads;
    }
}
