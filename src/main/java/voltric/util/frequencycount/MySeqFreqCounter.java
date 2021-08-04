package voltric.util.frequencycount;

import voltric.data.DiscreteData;
import voltric.data.DiscreteDataInstance;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class MySeqFreqCounter {

    public  static Map<Integer, Integer> compute(DiscreteData data, List<Integer> orderedVariableIndexes) {
        return computeJointCounts(data, orderedVariableIndexes, 0, data.getInstances().size());
    }

    // Por ahora solo computaria las joint counts (creo que es suficiente)
    private static Map<Integer, Integer> computeJointCounts(DiscreteData data, List<Integer> orderedVariableIndexes, int start, int end) {
        Map<Integer, Integer> jointCounts = new HashMap<>();

        List<DiscreteDataInstance> cases = data.getInstances();

        int[] allStates;
        int caseWeight;
        int[] projectedStates = new int[orderedVariableIndexes.size()];
        int projectedCaseWeight;
        int projectedCaseHashCode;

        for (int caseIndex = start; caseIndex < end; caseIndex++) {
            allStates = cases.get(caseIndex).getNumericValues();
            caseWeight = data.getWeight(cases.get(caseIndex));

            /** Rellenamos el vector de estados proyectados */
            for(int i = 0; i < orderedVariableIndexes.size(); i++){
                projectedStates[i] = allStates[orderedVariableIndexes.get(i)];
            }

            projectedCaseHashCode = Arrays.hashCode(projectedStates);

            if(!jointCounts.containsKey(projectedCaseHashCode)) {
                jointCounts.put(projectedCaseHashCode, caseWeight);
            }else {
                projectedCaseWeight = jointCounts.get(projectedCaseHashCode);
                projectedCaseWeight += caseWeight;
                jointCounts.put(projectedCaseHashCode, projectedCaseWeight);
            }
        }
        return jointCounts;
    }

    public static Map<Integer, Double> computeJointFrequencies(DiscreteData data, List<Integer> orderedVariableIndexes, int start, int end) {
        Map<Integer, Double> jointFrequencies = new HashMap<>();

        List<DiscreteDataInstance> cases = data.getInstances();

        int[] allStates;
        int caseWeight;
        int[] projectedStates = new int[orderedVariableIndexes.size()];
        double projectedCaseFreq;
        int projectedCaseHashCode;
        double totalWeight = (double) data.getTotalWeight(); // Le hacemos casting para que int/double = double

        for (int caseIndex = start; caseIndex < end; caseIndex++) {
            allStates = cases.get(caseIndex).getNumericValues();
            caseWeight = data.getWeight(cases.get(caseIndex));

            /** Rellenamos el vector de estados proyectados */
            for(int i = 0; i < orderedVariableIndexes.size(); i++){
                projectedStates[i] = allStates[orderedVariableIndexes.get(i)];
            }

            projectedCaseHashCode = Arrays.hashCode(projectedStates);

            if(!jointFrequencies.containsKey(projectedCaseHashCode)) {
                jointFrequencies.put(projectedCaseHashCode, caseWeight/totalWeight);
            }else {
                projectedCaseFreq = jointFrequencies.get(projectedCaseHashCode);
                projectedCaseFreq += caseWeight/totalWeight;
                jointFrequencies.put(projectedCaseHashCode, projectedCaseFreq);
            }
        }
        return jointFrequencies;
    }


    // TODO: Las variables tienen que estar ordenadas segun el dataSet?
    // TODO: Si divido entre el total Weight creo que pasarian a ser frecuencias y asi tendria sentido el nombre
/*
    public ArrayList<double[]> computeCounts(DiscreteData data, List<Integer> variableIndexes, int start, int end) {

        // the diagonal entries contain the frequencies of a single variable
        double[][] counts = new double[variableIndexes.size()][variableIndexes.size()];

        List<DiscreteDataInstance> cases = data.getInstances();

        for (int caseIndex = start; caseIndex < end; caseIndex++) {
            DiscreteDataInstance c = cases.get(caseIndex);
            int[] states = c.getNumericValues();
            double weight = data.getWeight(c);

            for(int i = 0; i < variableIndexes.size(); i++) {
                int vi = variableIndexes.get(i);
                for(int j = 0; j < variableIndexes.size(); j++) {
                    int vj = variableIndexes.get(j);

                    counts[i][j] += weight;

                }
            }


            // update the single and joint counts
            for (int i : entries) {
                //	int iInVariables = idMappingFromDataToVariables[i];
                int iInVariables = i;
                if (iInVariables < 0)
                    continue;

                for (int j : entries) {
                    //int jInVariables = idMappingFromDataToVariables[j];
                    int jInVariables = j;
                    if (jInVariables < 0)
                        continue;


                    double freq = frequencies.get(iInVariables)[jInVariables];
                    freq += weight;
                    frequencies.get(iInVariables)[jInVariables]=freq;

                }
            }
        }

    }
*/
}
