package deep.learning.C5;

import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import deep.learning.common.Util;

public class C5_5_活性化関数レイヤの実装 {

    @Test
    public void C5_5_1_ReLUレイヤ() {
        INDArray x = Nd4j.create(new double[][] {{1.0, -0.5}, {-2.0, 3.0}});
        assertEquals("[[1.00,-0.50],[-2.00,3.00]]", Util.string(x));
        // 本書とは違ったテストをします。
        Relu relu = new Relu();
        INDArray a = relu.forward(x);
        // forwardの結果
        assertEquals("[[1.00,0.00],[0.00,3.00]]", Util.string(a));
        // mask
        assertEquals("[[1.00,0.00],[0.00,1.00]]", Util.string(relu.mask));
        INDArray dout = Nd4j.create(new double[][] {{5, 6}, {7, 8}});
        INDArray b = relu.backward(dout);
        // backwardの結果
        assertEquals("[[5.00,0.00],[0.00,8.00]]", Util.string(b));
    }

    @Test
    public void C5_5_2_Sigmoidレイヤ() {
        // TODO: Sigmoidレイヤのテスト追加
    }
}
