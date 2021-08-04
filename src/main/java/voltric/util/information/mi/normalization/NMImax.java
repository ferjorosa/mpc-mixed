package voltric.util.information.mi.normalization;

import voltric.data.DiscreteData;
import voltric.inference.CliqueTreePropagation;
import voltric.model.DiscreteBayesNet;
import voltric.util.information.entropy.BnFactorizationEntropy;
import voltric.util.information.entropy.BnInferenceEntropy;
import voltric.util.information.entropy.FrequencyCountedEntropy;
import voltric.variables.DiscreteVariable;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Created by equipo on 25/07/2017.
 */
public class NMImax implements NormalizationFactor {

    @Override
    public double partialNormalizeMI(double mi, double px, double py, double pxy) {
        double xPartialEntropy = FrequencyCountedEntropy.computePartialValue(px);
        double yPartialEntropy = FrequencyCountedEntropy.computePartialValue(py);
        return mi /(Math.max(xPartialEntropy, yPartialEntropy));
    }

    @Override
    public double normalizeMI(double mi, List<DiscreteVariable> x, List<DiscreteVariable> y, DiscreteData data) {

        if(!data.getVariables().containsAll(x) || !data.getVariables().containsAll(y))
            throw new IllegalArgumentException("The DataSet must contain all the variables of X & Y");

        DiscreteData dataX = data.project(x);
        DiscreteData dataY = data.project(y);

        double xEntropy = FrequencyCountedEntropy.compute(dataX);
        double yEntropy = FrequencyCountedEntropy.compute(dataY);
        return mi / (Math.max(xEntropy, yEntropy));
    }

    // TODO: Repasar una vez se haya modificado la clase de DiscreteData
    @Override
    public double normalizeCMI(double cmi, List<DiscreteVariable> x, List<DiscreteVariable> y, List<DiscreteVariable> condVars, DiscreteData data) {

        // Ensure that the dataSet contains all the variables that compose X & Y & condVars
        if(!data.getVariables().containsAll(x) || !data.getVariables().containsAll(y) || !data.getVariables().containsAll(condVars))
            throw new IllegalArgumentException("The DataSet must contain all the variables of X & Y & condVars");

        Set<DiscreteVariable> xz = new LinkedHashSet<>();
        xz.addAll(x);
        xz.addAll(condVars);

        Set<DiscreteVariable> yz = new LinkedHashSet<>();
        yz.addAll(y);
        yz.addAll(condVars);

        DiscreteData dataXZ = data.project(new ArrayList<>(xz));
        DiscreteData dataYZ = data.project(new ArrayList<>(yz));

        double Hx_z = FrequencyCountedEntropy.computeConditional(condVars, dataXZ);
        double Hy_z = FrequencyCountedEntropy.computeConditional(condVars, dataYZ);

        return cmi / (Math.max(Hx_z, Hy_z));
    }

    /******************************************************************************************************************/

    @Override
    public double normalizeMI(double mi, DiscreteBayesNet xBn, DiscreteBayesNet yBn, DiscreteBayesNet xyBn) {

        double Hx = BnFactorizationEntropy.compute(xBn);
        double Hy = BnFactorizationEntropy.compute(yBn);

        return mi / (Math.max(Hx, Hy));
    }

    /******************************************************************************************************************/
    @Override
    public double normalizeMI(double mi, CliqueTreePropagation xCtp, CliqueTreePropagation yCtp, CliqueTreePropagation xyCtp, DiscreteData data) {
        double Hx = BnInferenceEntropy.compute(xCtp, data);
        double Hy = BnInferenceEntropy.compute(yCtp, data);

        return mi / (Math.max(Hx, Hy));
    }
}
