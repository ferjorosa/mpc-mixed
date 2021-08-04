package voltric.util.information.mi;

import voltric.data.DiscreteData;
import voltric.inference.CliqueTreePropagation;
import voltric.util.information.entropy.BnInferenceEntropy;

/**
 * Created by equipo on 14/11/2017.
 */
public class BnInferenceMI {

    public static double computePairwise(CliqueTreePropagation xCtp, CliqueTreePropagation yCtp, CliqueTreePropagation xyCtp, DiscreteData data) {

        double Hx = BnInferenceEntropy.compute(xCtp, data);
        double Hy = BnInferenceEntropy.compute(yCtp, data);
        double Hxy = BnInferenceEntropy.compute(xyCtp, data);

        return Hx + Hy - Hxy;
    }
}
