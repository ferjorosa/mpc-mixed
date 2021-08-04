package voltric.util.information.mi;

import voltric.data.DiscreteData;
import voltric.inference.CliqueTreePropagation;
import voltric.model.DiscreteBayesNet;
import voltric.potential.Function;
import voltric.variables.DiscreteVariable;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Esta clase actua como fachada de los metodos
 *
 * TODO: Revisar los metodos
 * TODO: IMPORTANTE, todavia queda todo el tema del apartado de analisis y las CMI y combinaciones de MI
 *
 * TODO: No puedo utilizar symmetric porque los edges son siempre dirigidos, ncluso en UndirectedGraph
 */
// TODO: Eliminar esta clase y dejar unicamente las versiones especificas de FreqCount y BNFact

public class MI {

    /** Using frequency-counted PDF, doesnt take into consideration conditional dependencies */

    public static double computePairwise(DiscreteVariable x, DiscreteVariable y, DiscreteData dataSet){
        return OldFrequencyCountedMI.computePairwise(x, y, dataSet);
    }

    public static double computePairwise(DiscreteVariable x, List<DiscreteVariable> y, DiscreteData dataSet){
        List<DiscreteVariable> oneVariableList = new ArrayList<>();
        oneVariableList.add(x);
        return MI.computePairwise(oneVariableList, y, dataSet);
    }

    public static double computePairwise(List<DiscreteVariable> x, List<DiscreteVariable> y, DiscreteData dataSet){
        return OldFrequencyCountedMI.computePairwise(x, y, dataSet);
    }

    public static double computeConditional(DiscreteVariable x, DiscreteVariable y, DiscreteVariable condVar, DiscreteData data){
        return FrequencyCountedMI.computeConditional(x, y, condVar, data);
    }

    public static double computeConditional(DiscreteVariable x, DiscreteVariable y, List<DiscreteVariable> condVars, DiscreteData data){
        return FrequencyCountedMI.computeConditional(x, y, condVars, data);
    }

    public static double computeConditional(List<DiscreteVariable> x, List<DiscreteVariable> y, List<DiscreteVariable> condVars, DiscreteData data){
        return FrequencyCountedMI.computeConditional(x, y, condVars, data);
    }

    /** Using the BN factorization for the Joint probability distribution */

    public static double computePairwise(DiscreteBayesNet xBn, DiscreteBayesNet yBn, DiscreteBayesNet xyBn) {
        return BnFactorizationMI.computePairwise(xBn, yBn, xyBn);
    }

    /** Using the BN inference method */

    public static double computePairwise(CliqueTreePropagation xCtp, CliqueTreePropagation yCtp, CliqueTreePropagation xyCtp, DiscreteData data) {
        return BnInferenceMI.computePairwise(xCtp, yCtp, xyCtp, data);
    }

    /** Using the JointDistribution method */

    public static double computePairwise(Function dist) {
        return JointDistributionMI.computePairwise(dist);
    }

    public static double computeConditional(Function dist, DiscreteVariable condVar) {
        return JointDistributionMI.computeConditional(dist, condVar);
    }

    /**********************************************************/
    /************************* EXTRA **************************/
    /**********************************************************/

    public static double computePairwiseParallel(DiscreteVariable x, DiscreteVariable y, DiscreteData dataSet){
        return OldFrequencyCountedMI.computePairwiseParallel(x, y, dataSet);
    }

    public static Map<DiscreteVariable, Map<DiscreteVariable, Double>> computePairwise(List<DiscreteVariable> variables, DiscreteData dataSet){
        return OldFrequencyCountedMI.computePairwise(variables, dataSet);
    }

    public static Map<DiscreteVariable, Map<DiscreteVariable, Double>> computePairwiseParallel(List<DiscreteVariable> variables, DiscreteData dataSet){
        return OldFrequencyCountedMI.computePairwiseParallel(variables, dataSet);
    }
}
