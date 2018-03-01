package deep.learning.C5;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Layer {

    INDArray forward(INDArray x);
    INDArray backward(INDArray x);

}
