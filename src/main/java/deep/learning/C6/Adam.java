package deep.learning.C6;

import org.nd4j.linalg.ops.transforms.Transforms;

import deep.learning.common.Optimizer;
import deep.learning.common.Params;

/**
 * class Adam:
 *     """Adam (http://arxiv.org/abs/1412.6980v8)"""
 *
 *     def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
 *         self.lr = lr
 *         self.beta1 = beta1
 *         self.beta2 = beta2
 *         self.iter = 0
 *         self.m = None
 *         self.v = None
 *
 *     def update(self, params, grads):
 *         if self.m is None:
 *             self.m, self.v = {}, {}
 *             for key, val in params.items():
 *                 self.m[key] = np.zeros_like(val)
 *                 self.v[key] = np.zeros_like(val)
 *
 *         self.iter += 1
 *         lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
 *
 *         for key in params.keys():
 *             #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
 *             #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
 *             self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
 *             self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
 *
 *             params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
 *
 *             #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
 *             #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
 *             #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)
 *
 */
public class Adam implements Optimizer {

    final double lr, beta1, beta2;
    int iter;
    Params m, v;

    public Adam(double lr, double beta1, double beta2) {
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.iter = 0;
    }

    public Adam(double lr) {
        this(lr, 0.9, 0.999);
    }

    public Adam() {
        this(0.001);
    }

    @Override
    public void update(Params params, Params grads) {
        if (m == null) {
            m = Params.zerosLike(params);
            v = Params.zerosLike(params);
        }
        ++iter;
        double lr_t = lr * Math.sqrt(1.0 - Math.pow(beta2, iter)) / (1.0 - Math.pow(beta1, iter));
        m.update((m, g) -> m.addi(g.sub(m).mul(1 - beta1)), grads);
        v.update((v, g) -> v.addi(g.mul(g).sub(v).mul(1 - beta2)), grads);
        params.update((p, m, v) -> p.subi(m.mul(lr_t).div(Transforms.sqrt(v).add(1e-7))), m, v);
    }

}
