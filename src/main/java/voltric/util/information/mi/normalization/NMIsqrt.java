package voltric.util.information.mi.normalization;

import org.apache.commons.lang3.NotImplementedException;
import voltric.data.DiscreteData;
import voltric.inference.CliqueTreePropagation;
import voltric.model.DiscreteBayesNet;
import voltric.util.information.entropy.BnFactorizationEntropy;
import voltric.util.information.entropy.BnInferenceEntropy;
import voltric.util.information.entropy.FrequencyCountedEntropy;
import voltric.variables.DiscreteVariable;

import java.util.List;

;

/**
 * Created by equipo on 25/07/2017.
 */
public class NMIsqrt implements NormalizationFactor {

    @Override
    public double partialNormalizeMI(double mi, double px, double py, double pxy) {
        double xPartialEntropy = FrequencyCountedEntropy.computePartialValue(px);
        double yPartialEntropy = FrequencyCountedEntropy.computePartialValue(py);
        return mi / (Math.sqrt(xPartialEntropy * yPartialEntropy));
    }

    @Override
    public double normalizeMI(double mi, List<DiscreteVariable> x, List<DiscreteVariable> y, DiscreteData dataSet) {

        DiscreteData dataX = dataSet.project(x);
        DiscreteData dataY = dataSet.project(y);

        double xEntropy = FrequencyCountedEntropy.compute(dataX);
        double yEntropy = FrequencyCountedEntropy.compute(dataY);
        return mi / (Math.sqrt(xEntropy * yEntropy));
    }

    @Override
    public double normalizeCMI(double cmi, List<DiscreteVariable> x, List<DiscreteVariable> y, List<DiscreteVariable> condVars, DiscreteData data) {
        throw new NotImplementedException("Nope");
    }

    /******************************************************************************************************************/

    @Override
    public double normalizeMI(double mi, DiscreteBayesNet xBn, DiscreteBayesNet yBn, DiscreteBayesNet xyBn) {

        double xEntropy = BnFactorizationEntropy.compute(xBn);
        double yEntropy = BnFactorizationEntropy.compute(yBn);

        return mi / (Math.sqrt(xEntropy * yEntropy));
    }

    /******************************************************************************************************************/
    @Override
    public double normalizeMI(double mi, CliqueTreePropagation xCtp, CliqueTreePropagation yCtp, CliqueTreePropagation xyCtp, DiscreteData data) {
        double Hx = BnInferenceEntropy.compute(xCtp, data);
        double Hy = BnInferenceEntropy.compute(yCtp, data);

        return mi / (Math.sqrt(Hx * Hy));
    }
}
