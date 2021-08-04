package voltric.util.information.mi.normalization;

import voltric.data.DiscreteData;
import voltric.inference.CliqueTreePropagation;
import voltric.model.DiscreteBayesNet;
import voltric.util.information.entropy.BnFactorizationEntropy;
import voltric.util.information.entropy.BnInferenceEntropy;
import voltric.util.information.entropy.FrequencyCountedEntropy;
import voltric.variables.DiscreteVariable;

import java.util.*;

/**
 * Created by equipo on 25/07/2017.
 */
public class NMIjoint implements NormalizationFactor {

    @Override
    public double partialNormalizeMI(double mi, double px, double py, double pxy) {
        double partialEntropy = FrequencyCountedEntropy.computePartialValue(pxy);
        return mi / partialEntropy;
    }

    @Override
    public double normalizeMI(double mi, List<DiscreteVariable> x, List<DiscreteVariable> y, DiscreteData data) {
        Set<DiscreteVariable> nonRepeatedSetOfVariables = new HashSet<>();
        nonRepeatedSetOfVariables.addAll(x);
        nonRepeatedSetOfVariables.addAll(y);

        // Ensure that the dataSet contains all the variables that compose X & Y
        if(!data.getVariables().containsAll(nonRepeatedSetOfVariables))
            throw new IllegalArgumentException("The DataSet must contain all the variables of X & Y");

        double jointEntropy = FrequencyCountedEntropy.compute(data.project(new ArrayList<>(nonRepeatedSetOfVariables))); // H(x,y)
        return mi / jointEntropy;
    }

    @Override
    public double normalizeCMI(double cmi, List<DiscreteVariable> x, List<DiscreteVariable> y, List<DiscreteVariable> condVars, DiscreteData data) {

        Set<DiscreteVariable> nonRepeatedSetOfVariables = new LinkedHashSet<>();
        nonRepeatedSetOfVariables.addAll(x);
        nonRepeatedSetOfVariables.addAll(y);
        nonRepeatedSetOfVariables.addAll(condVars);

        // Ensure that the dataSet contains all the variables that compose X & Y & condVars
        if(!data.getVariables().containsAll(nonRepeatedSetOfVariables))
            throw new IllegalArgumentException("The DataSet must contain all the variables of X & Y & condVars");

        double jointEntropy = FrequencyCountedEntropy.computeConditional(condVars, data.project(new ArrayList<>(nonRepeatedSetOfVariables))); // H(x,y|z)
        return cmi / jointEntropy;
    }

    /******************************************************************************************************************/
    @Override
    public double normalizeMI(double mi, DiscreteBayesNet xBn, DiscreteBayesNet yBn, DiscreteBayesNet xyBn) {

        double jointEntropy = BnFactorizationEntropy.compute(xyBn);
        return mi / jointEntropy;
    }

    /******************************************************************************************************************/
    @Override
    public double normalizeMI(double mi, CliqueTreePropagation xCtp, CliqueTreePropagation yCtp, CliqueTreePropagation xyCtp, DiscreteData data) {
        double jointEntropy = BnInferenceEntropy.compute(xyCtp, data);
        return mi / jointEntropy;
    }
}
