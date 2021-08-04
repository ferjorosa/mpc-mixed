package voltric.learning.structure.chowliu;

import voltric.data.DiscreteData;
import voltric.graph.UndirectedNode;
import voltric.graph.weighted.WeightedUndirectedGraph;
import voltric.inference.CliqueTreePropagation;
import voltric.model.DiscreteBayesNet;
import voltric.potential.Function;
import voltric.util.Tuple;
import voltric.util.stattest.discrete.DiscreteStatisticalTest;
import voltric.variables.DiscreteVariable;

import java.util.*;

/**
 * Este algoritmo deberia funcionar tanto para LVs como para MVs. Para comprobar su eficacia, voy a poner un metodo
 * extra de calcular la MI. En el caso de MVs probare que se obtenga la misma MI, ya qu teoricamente este metodo
 * seriviria para ambas posibilidades
 *
 * TODO: ESTA MAL IMPLEMENTADO, NO DEVUELVE EL CHOW-LIU, MIRAR VERSION NUEVA EN AMIDST
 */
@Deprecated
public class ChowLiu {

    // Este se utiliza en el Flat-LTM
    // TODO: Dado que el CL tree es no-dirigido, el statistical test deberia ser simétrico
    // TODO: NO PERMITE (DE MOMENTO) VARIABLES LATENTES, rehacerlo de forma similar a StatelessEmpDist
    @Deprecated
    public static WeightedUndirectedGraph<DiscreteVariable> learnChowLiuTree(List<DiscreteVariable> variables,
                                                                             DiscreteBayesNet bayesNet,
                                                                             DiscreteData dataSet,
                                                                             DiscreteStatisticalTest statisticalTest){

        for(DiscreteVariable variable: variables)
            if(!bayesNet.containsVar(variable))
                throw new IllegalArgumentException("All the variables need to belong to the Bayes net");

        for(DiscreteVariable variable: variables)
            if(variable.isManifestVariable() && !dataSet.getVariables().contains(variable))
                throw new IllegalArgumentException("All manifest variables need to be part of the DataSet");

        /**
         * 1 - El primer paso es obtener la informacion mutua entre todos los pares de variables
         */
        Map<Tuple<DiscreteVariable,DiscreteVariable>,Double> pairScores = new HashMap<>();
        int nVars = variables.size();

        // enumerate all variables
        for (int i = 0; i < nVars; i++) {
            DiscreteVariable vi = variables.get(i);
            for (int j = i + 1; j < nVars; j++) {
                DiscreteVariable vj = variables.get(j);
                Tuple<DiscreteVariable, DiscreteVariable> variablePair = new Tuple<>(vi, vj);

                // Compute empirical statistic test
                //TODO: No permite LVs, habria que modificar el codigo interno para que estimara sus posteriores (completar el DataSet)
                double pairValue = statisticalTest.computePairwise(vi, vj, dataSet);

                // The statistical test result should be symmetric, and it is stored as a symmetricPair
                pairScores.put(variablePair, pairValue);
            }
        }

        /**
         * 2 - Creamos un grafo completo y despues aplicamos el algoritmo de Prim para obtener el MaxWST
         */
        WeightedUndirectedGraph<DiscreteVariable> completeGraph = createCompleteGraph(pairScores, variables);
        Random randomGenerator = new Random();
        UndirectedNode<DiscreteVariable> startNode = completeGraph.getUndirectedNodes().get(randomGenerator.nextInt(completeGraph.getNumberOfNodes()));
        return completeGraph.maximumWeightSpanningTree(startNode);
    }

    // Este se utiliza en el TAN
    // TODO: Comprobar, sobretodo la generacion de la distribucion conjunta
    // TODO: Dado que el CL tree es no-dirigido, el statistical test deberia ser simétrico
    @Deprecated
    public static WeightedUndirectedGraph<DiscreteVariable> learnChowLiuTree(List<DiscreteVariable> variables,
                                                                             DiscreteVariable conditioningVar,
                                                                             DiscreteBayesNet bayesNet,
                                                                             DiscreteData dataSet,
                                                                             DiscreteStatisticalTest statisticalTest){
        for(DiscreteVariable variable: variables)
            if(!bayesNet.containsVar(variable))
                throw new IllegalArgumentException("All the variables need to belong to the Bayes net");

        for(DiscreteVariable variable: variables)
            if(variable.isManifestVariable() && !dataSet.getVariables().contains(variable))
                throw new IllegalArgumentException("All manifest variables need to be part of the DataSet");

        if(!bayesNet.containsVar(conditioningVar))
            throw new IllegalArgumentException("The conditioning variable must belong to the Bayes net");

        if(variables.contains(conditioningVar))
            throw new IllegalArgumentException("The conditioning variable cannot form part of the variables collection");

        /**
         * 1 - El primer paso es obtener la informacion mutua condicionada entre cada par de variables con respecto a la
         * variable condicionante.
         */
        //TODO: No se si es simetrica, hay que comprobarlo porque sino no valdria crear el completeGraph
        Map<Tuple<DiscreteVariable,DiscreteVariable>,Double> pairScores = new HashMap<>();

        int nVars = variables.size();

        // enumerate all variables
        for (int i = 0; i < nVars; i++) {
            DiscreteVariable vi = variables.get(i);
            for (int j = i + 1; j < nVars; j++) {
                DiscreteVariable vj = variables.get(j);
                Tuple<DiscreteVariable, DiscreteVariable> variablePair = new Tuple<>(vi, vj);

                //TODO: No permite LVs, habria que modificar el codigo interno para que estimara sus posteriores (completar el DataSet)
                double conditionalValue = statisticalTest.computeConditional(vi, vj, conditioningVar, dataSet);

                // The statistical test result should be symmetric, and it is stored as a symmetricPair
                pairScores.put(variablePair, conditionalValue);
            }
        }

        /**
         * 2 - Creamos un grafo completo y despues aplicamos el algoritmo de Prim para obtener el MaxWST
         */
        WeightedUndirectedGraph<DiscreteVariable> completeGraph = createCompleteGraph(pairScores, variables);
        Random randomGenerator = new Random();
        UndirectedNode<DiscreteVariable> startNode = completeGraph.getUndirectedNodes().get(randomGenerator.nextInt(completeGraph.getNumberOfNodes()));
        return completeGraph.maximumWeightSpanningTree(startNode);
    }

    public static WeightedUndirectedGraph<DiscreteVariable> learnTrueChowLiuTree(List<DiscreteVariable> manifestVariables,
                                                                                 DiscreteVariable conditioningVar,
                                                                                 DiscreteBayesNet bayesNet,
                                                                                 DiscreteData dataSet,
                                                                                 DiscreteStatisticalTest statisticalTest){
        for(DiscreteVariable variable: manifestVariables)
            if(!bayesNet.containsVar(variable))
                throw new IllegalArgumentException("All the variables need to belong to the Bayes net");

        for(DiscreteVariable variable: manifestVariables)
            if(variable.isManifestVariable() && !dataSet.getVariables().contains(variable))
                throw new IllegalArgumentException("All manifest variables need to be part of the DataSet");

        if(!bayesNet.containsVar(conditioningVar))
            throw new IllegalArgumentException("The conditioning variable must belong to the Bayes net");

        if(manifestVariables.contains(conditioningVar))
            throw new IllegalArgumentException("The conditioning variable cannot form part of the variables collection");

        /** Preparamos la inferencia y propagamos la evidencia (en este caso no hay pero es necesario llamar al metodo) */
        CliqueTreePropagation inferenceEngine = new CliqueTreePropagation(bayesNet);
        inferenceEngine.propagate();

        Map<Tuple<DiscreteVariable,DiscreteVariable>,Double> pairScores = new HashMap<>();

        int nVars = manifestVariables.size();

        // enumerate all variables
        for (int i = 0; i < nVars; i++) {
            DiscreteVariable vi = manifestVariables.get(i);
            for (int j = i + 1; j < nVars; j++) {
                DiscreteVariable vj = manifestVariables.get(j);
                Tuple<DiscreteVariable, DiscreteVariable> variablePair = new Tuple<>(vi, vj);
                List<DiscreteVariable> jointDistVars = new ArrayList<>();
                jointDistVars.add(vi);
                jointDistVars.add(vj);
                jointDistVars.add(conditioningVar);

                /** Calculamos la conjunta de las tres variables "vi", "vj" y "conditioningVar" */
                Function dist = inferenceEngine.computeBelief(jointDistVars);

                /** Calculamos el CMI or CNMI, etc (el correspondiente segun el statisticalTest) */
                double conditionalValue = statisticalTest.computeConditional(dist, conditioningVar);

                pairScores.put(variablePair, conditionalValue);
            }
        }

        /** 2 - Creamos un grafo completo y despues aplicamos el algoritmo de Prim para obtener el MaxWST */
        WeightedUndirectedGraph<DiscreteVariable> completeGraph = createCompleteGraph(pairScores, manifestVariables);
        Random randomGenerator = new Random();
        UndirectedNode<DiscreteVariable> startNode = completeGraph.getUndirectedNodes().get(randomGenerator.nextInt(completeGraph.getNumberOfNodes()));
        return completeGraph.maximumWeightSpanningTree(startNode);
    }

    private static WeightedUndirectedGraph<DiscreteVariable> createCompleteGraph(Map<Tuple<DiscreteVariable,DiscreteVariable>,Double> pairScores, List<DiscreteVariable> variables){
        WeightedUndirectedGraph<DiscreteVariable> weightedGraph = new WeightedUndirectedGraph<>();

        for(DiscreteVariable variable: variables)
            weightedGraph.addNode(variable);

        for(Tuple<DiscreteVariable, DiscreteVariable> variablePair: pairScores.keySet()) {
            UndirectedNode<DiscreteVariable> firstNode = weightedGraph.getNode(variablePair.getFirst());
            UndirectedNode<DiscreteVariable> secondNode = weightedGraph.getNode(variablePair.getSecond());
            weightedGraph.addEdge(firstNode, secondNode, pairScores.get(variablePair));
        }

        return weightedGraph;
    }
}
