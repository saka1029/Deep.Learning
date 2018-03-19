package deep.learning.common;

import java.io.Serializable;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.BiConsumer;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Params implements Serializable {

    private static final long serialVersionUID = 1L;

    private final Map<String, INDArray> params = new LinkedHashMap<>();

    /**
     * 指定したパラメータと同じキー、同じサイズで
     * 要素がすべてゼロで初期化されたパラメータを作成します。
     * @param origin 元になるパラメータを指定します。
     * @return
     */
    public static Params zerosLike(Params origin) {
        Params result = new Params();
        for (String key : origin.params.keySet())
            result.params.put(key, Nd4j.zerosLike(origin.get(key)));
        return result;
    }

    public INDArray get(String key) {
        return params.get(key);
    }

    public Params put(String key, INDArray value) {
        params.put(key, value);
        return this;
    }

    public int size() {
        return params.size();
    }

    /**
     *
     * @param updater 各配列を更新するラムダ式を指定します。
     *                このラムダ式は先頭の引数を更新するように記述する必要があります。
     *                たとえば<code>(p, a) -> p.addi(a)</code>などです。
     *                <code>(p, a) -> p.add(a)</code>はpの値を更新していないので誤りです。
     *                ラムダ式を評価した結果の値は単に捨てられる点に注意してください。
     * @param args
     */
    public void update(BiConsumer<INDArray, INDArray> updater, Params args) {
        for (String key : params.keySet())
            updater.accept(params.get(key), args.params.get(key));
    }

    /**
     *
     * @param updater 各配列を更新するラムダ式を指定します。
     *                このラムダ式は先頭の引数を更新するように記述する必要があります。
     *                たとえば<code>(p, a, b) -> p.addi(a.mul(b))</code>などです。
     *                <code>(p, a, b) -> p.add(a.mul(b))</code>はpの値を更新していないので誤りです。
     *                ラムダ式を評価した結果の値は単に捨てられる点に注意してください。
     * @param args0
     * @param args1
     */
    public void update(TriConsumer<INDArray> updater, Params args0, Params args1) {
        for (String key : params.keySet())
            updater.accept(params.get(key), args0.params.get(key), args1.params.get(key));
    }

    @Override
    public String toString() {
        return params.toString();
    }

}
