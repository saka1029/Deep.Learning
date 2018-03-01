package deep.learning.C5;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface LastLayer {

    double forward(INDArray x, INDArray t);
    INDArray backward(INDArray x);

}