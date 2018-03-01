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

import deep.learning.common.Functions;
import deep.learning.common.INDArrayFunction;
import deep.learning.common.TwoLayerParams;

public class TwoLayerNet {

    final TwoLayerParams parms;
    final Map<String, Layer> layers;
    final LastLayer lastLayer;

    public TwoLayerNet(int input_size, int hidden_size,
        int output_size) throws Exception {
        this(input_size, hidden_size, output_size, 0.01D);
    }

    public TwoLayerNet(int input_size, int hidden_size,
        int output_size, double weight_init_std) throws Exception {
        // 重みの初期化
        try (Random r = new DefaultRandom()) {
            parms = new TwoLayerParams(
                r.nextGaussian(new int[] {input_size, hidden_size}).mul(weight_init_std),
                Nd4j.zeros(hidden_size),
                r.nextGaussian(new int[] {hidden_size, output_size}).mul(weight_init_std),
                Nd4j.zeros(output_size));
        }
        // レイヤの生成
        layers = new LinkedHashMap<>();
        layers.put("Affine1", new Affine(parms.W1, parms.b1));
        layers.put("Relu1", new Relu());
        layers.put("Affine2", new Affine(parms.W2, parms.b2));
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
        return accuracy;
    }

    public TwoLayerParams numerical_gradient(INDArray x, INDArray t) {
        INDArrayFunction loss_W = W -> loss(x, t);
        TwoLayerParams grads = new TwoLayerParams(
            Functions.numerical_gradient(loss_W, parms.W1),
            Functions.numerical_gradient(loss_W, parms.b1),
            Functions.numerical_gradient(loss_W, parms.W2),
            Functions.numerical_gradient(loss_W, parms.b2));
        return grads;
    }

    public TwoLayerParams gradient(INDArray x, INDArray t) {
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
        TwoLayerParams grads = new TwoLayerParams(
            ((Affine)this.layers.get("Affine1")).dW,
            ((Affine)this.layers.get("Affine1")).db,
            ((Affine)this.layers.get("Affine2")).dW,
            ((Affine)this.layers.get("Affine2")).db);
        return grads;
    }
}
