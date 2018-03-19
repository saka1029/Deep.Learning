package deep.learning.C6;

import org.nd4j.linalg.ops.transforms.Transforms;

import deep.learning.common.Optimizer;
import deep.learning.common.Params;

public class AdaGrad implements Optimizer {

    final double lr;
    Params h;

    public AdaGrad(double lr) {
        this.lr = lr;
    }

    public AdaGrad() {
        this(0.01);
    }

    @Override
    public void update(Params params, Params grads) {
        if (h == null)
            h = Params.zerosLike(params);
        h.update((h, g) -> h.addi(g.mul(g)), grads);
        params.update((p, g, h) -> p.subi(g.mul(lr).div(Transforms.sqrt(h).add(1e-7))), grads, h);
    }
}
