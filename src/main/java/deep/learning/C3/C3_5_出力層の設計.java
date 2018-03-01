package deep.learning.C3;

import static org.junit.Assert.*;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import deep.learning.common.Util;

public class C3_5_出力層の設計 {

    @Test
    public void C3_5_1_恒等関数とソフトマックス関数() {
        INDArray a = Nd4j.create(new double[] {0.3, 2.9, 4.0});
        // 指数関数
        INDArray exp_a = Transforms.exp(a);
        assertEquals("[1.35,18.17,54.60]", Util.string(exp_a));
        // 指数関数の和
        Number sum_exp_a = exp_a.sumNumber();
        assertEquals(74.1221542102, sum_exp_a.doubleValue(), 5e-6);
        // ソフトマックス関数
        INDArray y = exp_a.div(sum_exp_a);
        assertEquals("[0.02,0.25,0.74]", Util.string(y));
    }

    public static INDArray softmax_wrong(INDArray a) {
        INDArray exp_a = Transforms.exp(a);
        Number sum_exp_a = exp_a.sumNumber();
        INDArray y = exp_a.div(sum_exp_a);
        return y;
    }

    public static INDArray softmax_right(INDArray a) {
        Number c = a.maxNumber();
        INDArray exp_a = Transforms.exp(a.sub(c));
        Number sum_exp_a = exp_a.sumNumber();
        INDArray y = exp_a.div(sum_exp_a);
        return y;
    }

    @Test
    public void C3_5_2_ソフトマックス関数実装上の注意() {
        INDArray a = Nd4j.create(new double[] {1010, 1000, 990});
        // 正しく計算されない
        assertEquals("[NaN,NaN,NaN]", Util.string(Transforms.exp(a).div(Transforms.exp(a).sumNumber())));
        Number c = a.maxNumber();
        assertEquals("[0.00,-10.00,-20.00]", Util.string(a.sub(c)));
        assertEquals("[1.00,0.00,0.00]", Util.string(Transforms.exp(a.sub(c)).div(Transforms.exp(a.sub(c)).sumNumber())));

        // 間違い
        assertEquals("[NaN,NaN,NaN]", Util.string(softmax_wrong(a)));
        // 正しい
        assertEquals("[1.00,0.00,0.00]", Util.string(softmax_right(a)));
        // ND4Jには正しいsoftmax(INDArray)が用意されています。
        assertEquals("[1.00,0.00,0.00]", Util.string(Transforms.softmax(a)));
    }

    @Test
    public void C3_5_4_ソフトマックス関数の特徴() {
        INDArray a = Nd4j.create(new double[] {0.3, 2.9, 4.0});
        INDArray y = Transforms.softmax(a);
        assertEquals("[0.02,0.25,0.74]", Util.string(y));
        // 総和は1になります。
        assertEquals(1.0, y.sumNumber().doubleValue(), 5e-6);
    }

}
