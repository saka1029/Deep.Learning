package deep.learning.common;

import org.nd4j.linalg.api.ndarray.INDArray;

public class TwoLayerParams {

    public final INDArray W1, b1, W2, b2;

    public TwoLayerParams(INDArray W1, INDArray b1, INDArray W2, INDArray b2) {
        this.W1 = W1;
        this.b1 = b1;
        this.W2 = W2;
        this.b2 = b2;
    }

    @Override
    public String toString() {
        return String.format("W1=%s,b1=%s,W2=%s,b2=%s", W1, b1, W2, b2);
    }
}
