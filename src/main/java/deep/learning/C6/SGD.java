package deep.learning.C6;

import deep.learning.common.Optimizer;
import deep.learning.common.Params;

public class SGD implements Optimizer {

    /** learning rate (学習係数) */
    final double lr;

    public SGD(double lr) {
        this.lr = lr;
    }

    public SGD() {
        this(0.01);
    }

    @Override
    public void update(Params params, Params grads) {
        params.update((p, g) -> p.subi(g.mul(lr)), grads);
    }
}
