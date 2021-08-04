package voltric.learning.structure.incremental.util;

import voltric.data.DiscreteData;
import voltric.data.DiscreteDataInstance;
import voltric.inference.CliqueTreePropagation;
import voltric.model.DiscreteBayesNet;
import voltric.potential.Function;
import voltric.util.information.entropy.FrequencyCountedEntropy;
import voltric.variables.DiscreteVariable;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class IncLearnerUtils {

    public static int[] predictData(DiscreteData data, DiscreteBayesNet bayesNet, String varName) {

        DiscreteVariable predictVariable = bayesNet.getNode(varName).getVariable();
        int[] predictionValues = new int[data.getInstances().size()];

        CliqueTreePropagation inferenceEngine = new CliqueTreePropagation(bayesNet);
        Map<DiscreteVariable, Integer> evidence = new HashMap<>();
        for(int instIndex = 0; instIndex < data.getInstances().size(); instIndex++) {
            DiscreteDataInstance dataCase = data.getInstances().get(instIndex);
            for (int i = 0; i < dataCase.getVariables().size(); i++)
                evidence.put(data.getVariables().get(i), dataCase.getNumericValue(i));
            inferenceEngine.setEvidence(evidence);
            // Propagate the evidence and compute the belief for the predict variable
            inferenceEngine.propagate();
            Function predictVariableFunc = inferenceEngine.computeBelief(predictVariable);
            int indexMaxProbability = maxProbIndex(predictVariableFunc.getCells());
            predictionValues[instIndex] = indexMaxProbability;
        }

        return predictionValues;
    }

    public static double mi(DiscreteVariable x, DiscreteVariable y, DiscreteData data) {

        List<DiscreteVariable> variableList = new ArrayList<>(2);
        variableList.add(x);
        variableList.add(y);

        DiscreteData xyData = data.projectV3(variableList);
        double Hxy = FrequencyCountedEntropy.compute(xyData);

        List<DiscreteVariable> xList = new ArrayList<>(1);
        xList.add(x);
        DiscreteData xData = data.projectV3(xList);
        double Hx = FrequencyCountedEntropy.compute(xData);

        List<DiscreteVariable> yList = new ArrayList<>(1);
        xList.add(y);
        DiscreteData yData = data.projectV3(yList);
        double Hy = FrequencyCountedEntropy.compute(yData);

        return Hx + Hy - Hxy;
    }

    public static double mi(int[][] data, boolean normalization) {
        /* Create the DiscreteDataSet, which will make it easier to estimate the MI */
        VoltricDiscreteDataSet xyCountsData = new VoltricDiscreteDataSet(data);

        Map<Integer, Double> xFreqs= new HashMap<>();
        Map<Integer, Double> yFreqs= new HashMap<>();
        Map<int[], Double> xyFreqs = new HashMap<>();

        /* Frequencies estimation */
        int N = data.length;
        for(int[] instance: xyCountsData) {
            int instanceCount = xyCountsData.getCounts(instance);
            double instanceFreq = (double) instanceCount / N;

            double xFreq = 0;
            if(xFreqs.containsKey(instance[0]))
                xFreq = xFreqs.get(instance[0]);

            double yFreq = 0;
            if(yFreqs.containsKey(instance[1]))
                yFreq = yFreqs.get(instance[1]);

            xFreqs.put(instance[0], xFreq + instanceFreq);
            yFreqs.put(instance[1], yFreq + instanceFreq);
            xyFreqs.put(instance, instanceFreq);
        }

        /* Estimate entropies from frequencies */
        double Hx = 0;
        for(int instanceX: xFreqs.keySet()){
            double freq = xFreqs.get(instanceX);
            Hx -= freq * Math.log(freq);
        }

        double Hy = 0;
        for(int instanceY: yFreqs.keySet()){
            double freq = yFreqs.get(instanceY);
            Hy -= freq * Math.log(freq);
        }

        double Hxy = 0;
        for(int[] instance: xyFreqs.keySet()){
            double freq = xyFreqs.get(instance);
            Hxy -= freq * Math.log(freq);
        }

        double mi = Hx + Hy - Hxy;
        double normalizationFactor = Math.min(Hx, Hy);

        if(normalization)
            return mi / normalizationFactor;

        return mi;
    }

    private static int maxProbIndex(double[] probs) {
        int index = 0;
        for(int i = 0; i < probs.length; i++)
            if(probs[index] < probs[i])
                index = i;

        return index;
    }
}
