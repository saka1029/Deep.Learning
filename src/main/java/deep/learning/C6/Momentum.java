package deep.learning.C6;

import deep.learning.common.Optimizer;
import deep.learning.common.Params;

public class Momentum implements Optimizer {

    final double lr, momentum;
    Params v;

    public Momentum(double lr, double momentum) {
        this.lr = lr;
        this.momentum = momentum;
        this.v = null;
    }

    public Momentum(double lr) {
        this(lr, 0.9);
    }

    public Momentum() {
        this(0.01);
    }

    @Override
    public void update(Params params, Params grads) {
        if (v == null)
            v = Params.zerosLike(params);
        v.update((v, g) -> v.muli(momentum), grads);
        v.update((v, g) -> v.subi(g.mul(lr)), grads);
        params.update((p, v) -> p.addi(v), v);
    }
}