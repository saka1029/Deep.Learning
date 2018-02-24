package deep.learning.common;

import org.nd4j.linalg.api.ndarray.INDArray;

@FunctionalInterface
public interface INDArrayFunction {
    double apply(INDArray a);
}

