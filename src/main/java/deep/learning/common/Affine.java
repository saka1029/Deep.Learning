package deep.learning.common;

import org.nd4j.linalg.api.ndarray.INDArray;

public class Affine implements Layer {

    public final INDArray W, b;
    INDArray x;
    public INDArray dW, db;

    public Affine(INDArray W, INDArray b) {
        this.W = W;
        this.b = b;
    }

    @Override
    public INDArray forward(INDArray x) {
        this.x = x;
        return x.mmul(W).addRowVector(b);
    }

    @Override
    public INDArray backward(INDArray dout) {
        // W.transpose()はWの転置行列です。
        INDArray dx = dout.mmul(W.transpose());
        dW = x.transpose().mmul(dout);
        // 集約して1行にします。
        db = dout.sum(0);
        return dx;
    }
}
