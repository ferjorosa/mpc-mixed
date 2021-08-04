package voltric.util.information.mi;

import voltric.model.DiscreteBayesNet;
import voltric.util.information.entropy.BnFactorizationEntropy;

/**
 * TODO: En un futuro habria que ver cual es la implementacion de la MI mas optima en tiempo de computacion
 */
public class BnFactorizationMI {

    public static double computePairwise(DiscreteBayesNet xBn, DiscreteBayesNet yBn, DiscreteBayesNet xyBn) {

        double Hx = BnFactorizationEntropy.compute(xBn);
        double Hy = BnFactorizationEntropy.compute(yBn);
        double Hxy = BnFactorizationEntropy.compute(xyBn);

        return Hx + Hy - Hxy;
    }
}
