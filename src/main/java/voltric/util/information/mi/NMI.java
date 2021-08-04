package voltric.util.information.mi;

import voltric.data.DiscreteData;
import voltric.inference.CliqueTreePropagation;
import voltric.model.DiscreteBayesNet;
import voltric.util.information.mi.normalization.NormalizationFactor;
import voltric.variables.DiscreteVariable;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Esta clase actua como fachada de los metodos
 *
 * TODO: La normalized conditional information no se ha implementado aun porque requiere revisar la teoria
 * Antes de meterme en la normalized CMI, seria mejor terminar la CMI
 */

// TODO: Eliminar esta clase y dejar unicamente las versiones especificas de FreqCount y BNFact
public class NMI {

    public static double computePairwise(DiscreteVariable x, DiscreteVariable y, DiscreteData dataSet, NormalizationFactor normalizationFactor){
        double mi = MI.computePairwise(x, y, dataSet);

        List<DiscreteVariable> xList = new ArrayList<>();
        List<DiscreteVariable> yList = new ArrayList<>();
        xList.add(x);
        yList.add(y);

        return normalizationFactor.normalizeMI(mi, xList, yList , dataSet);
    }

    public static double computePairwise(DiscreteVariable x, List<DiscreteVariable> y, DiscreteData dataSet, NormalizationFactor normalizationFactor){
        List<DiscreteVariable> oneVariableList = new ArrayList<>();
        oneVariableList.add(x);
        return NMI.computePairwise(oneVariableList, y, dataSet, normalizationFactor);
    }

    public static double computePairwise(List<DiscreteVariable> x, List<DiscreteVariable> y, DiscreteData dataSet, NormalizationFactor normalizationFactor){
        double mi = MI.computePairwise(x, y, dataSet);

        return normalizationFactor.normalizeMI(mi, x, y , dataSet);
    }

    public static double computeConditional(DiscreteVariable x, DiscreteVariable y, DiscreteVariable condVar, DiscreteData data, NormalizationFactor normalizationFactor){
        double cmi = FrequencyCountedMI.computeConditional(x, y, condVar, data);

        List<DiscreteVariable> xList = new ArrayList<>();
        List<DiscreteVariable> yList = new ArrayList<>();
        List<DiscreteVariable> condVars = new ArrayList<>();
        xList.add(x);
        yList.add(y);
        condVars.add(condVar);

        return normalizationFactor.normalizeCMI(cmi, xList, yList,condVars, data);
    }

    public static double computeConditional(DiscreteVariable x, DiscreteVariable y, List<DiscreteVariable> condVars, DiscreteData data, NormalizationFactor normalizationFactor){
        double cmi = FrequencyCountedMI.computeConditional(x, y, condVars, data);

        List<DiscreteVariable> xList = new ArrayList<>();
        List<DiscreteVariable> yList = new ArrayList<>();
        xList.add(x);
        yList.add(y);

        return normalizationFactor.normalizeCMI(cmi, xList, yList,condVars, data);
    }

    public static double computeConditional(List<DiscreteVariable> x, List<DiscreteVariable> y, List<DiscreteVariable> condVars, DiscreteData data, NormalizationFactor normalizationFactor){
        double cmi = FrequencyCountedMI.computeConditional(x, y, condVars, data);

        return normalizationFactor.normalizeCMI(cmi, x, y,condVars, data);
    }

    /********************************************** BN FACTORIZATION **************************************************/

    public static double computePairwise(DiscreteBayesNet xBn, DiscreteBayesNet yBn, DiscreteBayesNet xyBn, NormalizationFactor normalizationFactor) {
        double mi = BnFactorizationMI.computePairwise(xBn, yBn, xyBn);
        return normalizationFactor.normalizeMI(mi, xBn, yBn, xyBn);
    }

    /********************************************** BN INFERENCE **************************************************/

    public static double computePairwise(CliqueTreePropagation xCtp, CliqueTreePropagation yCtp, CliqueTreePropagation xyCtp, DiscreteData data, NormalizationFactor normalizationFactor) {
        double mi = BnInferenceMI.computePairwise(xCtp, yCtp, xyCtp, data);
        return normalizationFactor.normalizeMI(mi, xCtp, yCtp, xyCtp, data);
    }

    /********************************************* EXTRA ******************************************/

    public static Map<DiscreteVariable, Map<DiscreteVariable, Double>> computePairwise(List<DiscreteVariable> variables, DiscreteData dataSet, NormalizationFactor normalizationFactor){
        Map<DiscreteVariable, Map<DiscreteVariable, Double>> miMap = MI.computePairwise(variables, dataSet);

        // Each value is normalized
        for(DiscreteVariable firstVar: miMap.keySet())
            for(DiscreteVariable secondVar: miMap.get(firstVar).keySet()) {
                // The "normalizeMI" method needs to receive a List<DiscreteVariable> as argument
                List<DiscreteVariable> firstVarList = new ArrayList<>();
                List<DiscreteVariable> secondVarList = new ArrayList<>();
                firstVarList.add(firstVar);
                secondVarList.add(secondVar);

                miMap.get(firstVar).put(secondVar, normalizationFactor.normalizeMI(miMap.get(firstVar).get(secondVar), firstVarList, secondVarList, dataSet));
            }

        // the normalized map is returned
        return miMap;
    }

    public static Map<DiscreteVariable, Map<DiscreteVariable, Double>> computePairwiseParallel(List<DiscreteVariable> variables, DiscreteData dataSet, NormalizationFactor normalizationFactor){
        Map<DiscreteVariable, Map<DiscreteVariable, Double>> miMap = MI.computePairwiseParallel(variables, dataSet);

        // Each value is normalized
        for(DiscreteVariable firstVar: miMap.keySet())
            for(DiscreteVariable secondVar: miMap.get(firstVar).keySet()) {
                // The "normalizeMI" method needs to receive a List<DiscreteVariable> as argument
                List<DiscreteVariable> firstVarList = new ArrayList<>();
                List<DiscreteVariable> secondVarList = new ArrayList<>();
                firstVarList.add(firstVar);
                secondVarList.add(secondVar);

                miMap.get(firstVar).put(secondVar, normalizationFactor.normalizeMI(miMap.get(firstVar).get(secondVar), firstVarList, secondVarList, dataSet));
            }

        // the normalized map is returned
        return miMap;
    }

    public static double computePairwiseParallel(DiscreteVariable x, DiscreteVariable y, DiscreteData dataSet, NormalizationFactor normalizationFactor){
        double mi = MI.computePairwiseParallel(x, y, dataSet);

        List<DiscreteVariable> xList = new ArrayList<>();
        List<DiscreteVariable> yList = new ArrayList<>();
        xList.add(x);
        yList.add(y);

        return normalizationFactor.normalizeMI(mi, xList, yList , dataSet);
    }


}
