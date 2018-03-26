package deep.learning.common;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class BatchNormalization implements BatchNormLayer {

    final INDArray gamma, beta;
    final double momentum;
    int[] input_shape;
    INDArray running_mean, running_var;
    int batch_size;
    INDArray xc, xn, std;
    public INDArray dgamma, dbeta;

    public BatchNormalization(INDArray gamma, INDArray beta,
        double momentum/*=0.9*/,
        INDArray running_mean/*=null*/, INDArray running_var/*=null*/) {
        this.gamma = gamma;
        this.beta = beta;
        this.momentum = 0.9;
        // テスト時に使用する平均と分散
        this.running_mean = running_mean;
        this.running_var = running_var;
        // backward時に使用する中間データ
    }

    public BatchNormalization(INDArray gamma, INDArray beta) {
        this(gamma, beta, 0.9, null, null);
    }

    @Override
    public INDArray forward(INDArray x, boolean train_flg) {
        input_shape = x.shape();
        // TODO: 実装方法が不明
//        if (x.size(0) == 1) {
//        }
        INDArray out = __forward(x, train_flg);
        return out.reshape(input_shape);
    }

    private INDArray __forward(INDArray x, boolean train_flg) {
        if (running_mean == null) {
            int D = x.size(1);
            running_mean = Nd4j.zeros(D);
            running_var = Nd4j.zeros(D);
        }
        INDArray xn;
        if (train_flg) {
            INDArray mu = x.mean(0);
            INDArray xc = x.subRowVector(mu);
            INDArray var = xc.mul(xc).mean(0);
            INDArray std = Transforms.sqrt(var.add(10e-7));
            xn = xc.divRowVector(std);

            this.batch_size = x.size(0);
            this.xc = xc;
            this.xn = xn;
            this.std = std;
            this.running_mean = running_mean.mul(momentum).add(mu.mul(1 - momentum));
            this.running_var = running_var.mul(momentum).add(var.mul(1 - momentum));
        } else {
            INDArray xc = x.subRowVector(running_mean);
            xn = xc.divRowVector(Transforms.sqrt(running_var.add(10e-7)));
        }
        INDArray out = xn.mulRowVector(gamma).addRowVector(beta);
        return out;
    }

    @Override
    public INDArray backward(INDArray dout) {
        // TODO: 実装方法不明
        if (dout.rank() != 2) {
        }
        INDArray dx = __backward(dout);
        dx = dx.reshape(input_shape);
        return dx;
    }

    private INDArray __backward(INDArray dout) {
        INDArray dbeta = dout.sum(0);
        INDArray dgamma = xn.mul(dout).sum(0);
        INDArray dxn = dout.mulRowVector(gamma);
        INDArray dxc = dxn.divRowVector(std);
        INDArray dstd = dxn.mul(xc).divRowVector(std.mul(std)).sum(0).neg();
        INDArray dvar = dstd.mul(0.5).div(std);
        dxc.addi(xc.mul(2.0 / batch_size).mulRowVector(dvar));
        INDArray dmu = dxc.sum(0);
        INDArray dx = dxc.subRowVector(dmu.div(batch_size));
        this.dgamma = dgamma;
        this.dbeta = dbeta;
        return dx;
    }

}
