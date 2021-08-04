package voltric.learning.structure.hillclimbing.efficientlocal;

import voltric.data.DiscreteData;
import voltric.data.DiscreteDataInstance;
import voltric.learning.score.ScoreType;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.potential.Function;
import voltric.util.Tuple;
import voltric.util.Utils;
import voltric.variables.DiscreteVariable;

import java.util.*;
import java.util.stream.Collectors;

public class EffLocalScore {

    public static Map<DiscreteVariable, Double> computeNetScores (DiscreteBayesNet bayesNet, DiscreteData data, ScoreType scoreType) {
        Map<DiscreteVariable, Double> scores = new HashMap<>();

        /** Iteramos por cada uno de los nodos de la BN */
        for(DiscreteBeliefNode node: bayesNet.getNodes()) {

            /** Obtenemos la familia del nodo en cuestion */
            List<DiscreteVariable> family = node.getDiscreteParentVariables();
            family.add(node.getVariable());

            /** Proyectamos los datos sobre las variables de la familia del nodo */
            DiscreteData projectedData = data.project(family);

            Function nodeCpt = node.getCpt();

            /** Iteramos por los datos proyectados */
            double nodeLL = 0;
            for(DiscreteDataInstance projectedDataInstance: projectedData.getInstances()) {
                double p = nodeCpt.getCells()[nodeCpt.computeIndex(projectedDataInstance.getNumericValues())];
                nodeLL += Utils.log(p)* projectedData.getWeight(projectedDataInstance);
            }

            double nodeScore = 0;
            switch (scoreType){
                case LogLikelihood:
                    nodeScore += nodeLL;
                    break;
                case BIC:
                    nodeScore += nodeLL - computeNewDimension(node.getVariable(), family) * Utils.log(projectedData.getTotalWeight()) / 2.0;
                    break;
                case AIC:
                    nodeScore += nodeLL - computeNewDimension(node.getVariable(), family);
                    break;
            }

            /** Almacenamos el score especifico */
            scores.put(node.getVariable(), nodeScore);
        }

        return scores;
    }

    public static double computeNodeScore(DiscreteVariable nodeVar,
                                            List<DiscreteVariable> nodeFamily,
                                            Function newCPT,
                                            ScoreType scoreType,
                                            DiscreteData projectedData) {
        double nodeLL = 0;
        for(DiscreteDataInstance projectedDataInstance: projectedData.getInstances()) {
            double p = newCPT.getCells()[newCPT.computeIndex(projectedDataInstance.getNumericValues())];
            nodeLL += Utils.log(p)* projectedData.getWeight(projectedDataInstance);
        }

        /** Calculate the node score */
        double nodeScore = 0;
        switch (scoreType){
            case LogLikelihood:
                nodeScore += nodeLL;
                break;
            case BIC:
                nodeScore += nodeLL - computeNewDimension(nodeVar, nodeFamily) * Utils.log(projectedData.getTotalWeight()) / 2.0;
                break;
            case AIC:
                nodeScore += nodeLL - computeNewDimension(nodeVar, nodeFamily);
                break;
        }

        return nodeScore;
    }

    public static double computeLocalScore(DiscreteVariable variable, List<DiscreteVariable> family, DiscreteData data, EfficientDiscreteData efficientDiscreteData, ScoreType scoreType) {

        /** Ordenamos las variables de la familia segun su indice */
        List<Tuple<Integer, DiscreteVariable>> orderedListWithIndex = sortAndPairWithOriginalIndex(family, data);
        List<DiscreteVariable> orderedFamily = orderedListWithIndex.stream().map(pair->pair.getSecond()).collect(Collectors.toList());
        List<Integer> orderedFamilyIndexes = orderedListWithIndex.stream().map(pair->pair.getFirst()).collect(Collectors.toList());

        /** Creamos la estructuras básica para iterar: magnitudes */
        int numberOfVars = orderedFamily.size();
        int[] magnitudes = new int[numberOfVars];
        int magnitude = 1;
        for (int i = numberOfVars - 1; i >= 0; i--) {
            magnitudes[i] = magnitude;
            magnitude *= orderedFamily.get(i).getCardinality();
        }

        /** Computamos las jointCounts con EfficientDiscreteData*/
        int domainSize = magnitude;// Tras la iteracion magnitude == domainSize
        int[] jointCounts = new int[domainSize];
        int[] projectedStates = new int[orderedFamily.size()];

        int[][] efficientDataContent = efficientDiscreteData.content;
        int[] efficientDataWeights = efficientDiscreteData.weights;

        for(int i = 0; i < efficientDataContent.length; i++){

            for(int j = 0; j < orderedFamilyIndexes.size(); j++)
                projectedStates[j] = efficientDataContent[i][orderedFamilyIndexes.get(j)];

            // Almacenamos el peso del dataCase en su luegar adecuado
            jointCounts[computeIndex(projectedStates, numberOfVars, magnitudes)] += efficientDataWeights[i];
        }
/*
        for (DiscreteDataInstance dataCase : data.getInstances()) {
            allStates = dataCase.getNumericValues();

            // Rellenamos el vector de estados proyectados
            for(int i = 0; i < orderedFamilyIndexes.size(); i++)
                projectedStates[i] = allStates[orderedFamilyIndexes.get(i)];

            // Almacenamos el peso del dataCase en su luegar adecuado
            jointCounts[computeIndex(projectedStates, numberOfVars, magnitudes)] += data.getWeight(dataCase);
        }
*/

        /** Normalizamos los jointCounts para obtener las freqs */
        double[] freqs;
        if(orderedFamily.size() == 1) {
            freqs = computeMarginalFreqs(jointCounts);
        }else {
            int nodeVarIndexInFamily = orderedFamily.indexOf(variable);
            freqs = computeCondFreqs(jointCounts, variable, nodeVarIndexInFamily, magnitudes);
        }

        /** Una vez calculadas las frecuencias, estimamos la likelihood */
        double logLikelihood = 0;
        for (int i = 0; i < jointCounts.length; i++)
            logLikelihood += jointCounts[i] * Utils.log(freqs[i]);

        /** Una vez tenemos la Log-Likelihood, computamos el score final */
        double nodeScore = 0;
        switch (scoreType){
            case LogLikelihood:
                nodeScore += logLikelihood;
                break;
            case BIC:
                nodeScore += logLikelihood - computeNewDimension(variable, orderedFamily) * Utils.log(data.getTotalWeight()) / 2.0;
                break;
            case AIC:
                nodeScore += logLikelihood - computeNewDimension(variable, orderedFamily);
                break;
        }

        return nodeScore;
    }

    private static List<Tuple<Integer, DiscreteVariable>> sortAndPairWithOriginalIndex(List<DiscreteVariable> variables, DiscreteData data){

        Comparator<Tuple<Integer, DiscreteVariable>> byIndex =
                (pair1, pair2) -> Integer.compare(pair1.getFirst(), pair2.getFirst());

        // Pseudo zip with index
        return data.getVariables().stream()
                .filter(variables::contains)
                .map(x-> new Tuple<>(data.getVariables().indexOf(x), x))
                .sorted(byIndex)
                .collect(Collectors.toCollection(ArrayList::new));
    }

    private static int computeIndex(int[] projectedStates, int numberOfVars, int[] magnitudes) {
        int index = 0;

        for (int i = 0; i < numberOfVars; i++) {
            index += (projectedStates[i] * magnitudes[i]);
        }

        return index;
    }

    private static double[] computeMarginalFreqs(int[] jointCounts) {
        double sum = 0.0;
        double[] freqs = new double[jointCounts.length];
        for(int i = 0; i < jointCounts.length; i++)
            sum += jointCounts[i];
        for(int i = 0; i < jointCounts.length; i++)
            freqs[i] = jointCounts[i] / sum;
        return freqs;
    }

    private static double[] computeCondFreqs(int[] jointCounts, DiscreteVariable variable, int variableIndex, int[] magnitudes) {
        int domainSize = jointCounts.length;
        double[] condFreqs = new double[domainSize];
        /** */
        int cardinality = variable.getCardinality();
        int subdomainSize = domainSize / cardinality;

        /** */
        int magnitude = magnitudes[variableIndex];
        int magnitude2 = magnitude * cardinality;
        int carry = 0;
        int residual = 0;

        /** */
        int[] affectedCells = new int[cardinality];

        for (int i = 0; i < subdomainSize; i++) {
            // computes the index in the original domain
            int index = carry + residual;

            // computes normalizing constant
            double sum = 0.0;
            for (int j = 0; j < cardinality; j++) {
                sum += jointCounts[index];
                affectedCells[j] = index;
                index += magnitude;
            }
            // TODO: Check si tiene sentido, creo que si, para que no se de el caso de dividir entre 0
            if (sum != 0.0) {
                for (int j = 0; j < cardinality; j++) {
                    condFreqs[affectedCells[j]] = jointCounts[affectedCells[j]] / sum;
                }
            }

            // next element in original domain
            residual++;

            if (residual == magnitude) {
                // carries in
                carry += magnitude2;
                residual = 0;
            }
        }

        return condFreqs;
    }

    private static int computeNewDimension(DiscreteVariable nodeVar, List<DiscreteVariable> family) {
        int dimension = nodeVar.getCardinality() - 1;
        for(DiscreteVariable var: family)
            if(!var.equals(nodeVar))
                dimension *= var.getCardinality();
        return dimension;
    }
}
