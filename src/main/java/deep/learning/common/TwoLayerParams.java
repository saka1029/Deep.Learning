package deep.learning.common;

import java.io.Serializable;

import org.nd4j.linalg.api.ndarray.INDArray;

public class TwoLayerParams implements Serializable {

    private static final long serialVersionUID = 3747469451731698544L;
    public final INDArray W1, b1, W2, b2;

    public TwoLayerParams(INDArray W1, INDArray b1, INDArray W2, INDArray b2) {
        this.W1 = W1;
        this.b1 = b1;
        this.W2 = W2;
        this.b2 = b2;
    }

    @Override
    public String toString() {
        return String.format("W1=%s%nb1=%s%nW2=%s%nb2=%s%n",
            Util.toStringExact(W1),
            Util.toStringExact(b1),
            Util.toStringExact(W2),
            Util.toStringExact(b2));
    }
}
