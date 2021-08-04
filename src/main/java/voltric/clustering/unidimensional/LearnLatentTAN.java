package voltric.clustering.unidimensional;


import voltric.data.DiscreteData;
import voltric.graph.AbstractNode;
import voltric.graph.DirectedAcyclicGraph;
import voltric.graph.DirectedNode;
import voltric.graph.Edge;
import voltric.graph.weighted.WeightedUndirectedGraph;
import voltric.learning.LearningResult;
import voltric.learning.parameter.DiscreteParameterLearning;
import voltric.learning.structure.chowliu.ChowLiu;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.model.HLCM;
import voltric.model.creator.HlcmCreator;
import voltric.util.stattest.discrete.DiscreteStatisticalTest;
import voltric.variables.DiscreteVariable;
import voltric.variables.Variable;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Nueva version del Hidden TAN basada en el structural EM
 */
public class LearnLatentTAN {

    public static LearningResult<DiscreteBayesNet> learnModel(int cardinality,
                                                              DiscreteData dataSet,
                                                              DiscreteParameterLearning parameterLearning,
                                                              DiscreteStatisticalTest statisticalTest,
                                                              double threshold) {
        return applySEM(cardinality, parameterLearning, dataSet, statisticalTest, threshold);
    }

    public static LearningResult<DiscreteBayesNet> learnModelToMaxCardinality(int maxCardinality,
                                                              DiscreteData dataSet,
                                                              DiscreteParameterLearning parameterLearning,
                                                              double threshold,
                                                              DiscreteStatisticalTest statisticalTest){

        /** 1 - Creamos un modelo inicial TAN cuya variable latente tiene cardinalidad 2 */
        int currentCardinality = 2;
        LearningResult<DiscreteBayesNet> initialModel = applySEM(currentCardinality, parameterLearning, dataSet, statisticalTest, threshold);
        LearningResult<DiscreteBayesNet> bestModel = initialModel;

        /** Aumentamos de forma sucesiva la cardinalidad de la LV aplicando SEM para mantener el mejor TAN hasta que el score deje de mejorar*/
        while(currentCardinality <= maxCardinality){
            currentCardinality += 1;

            LearningResult<DiscreteBayesNet> currentModel = applySEM(currentCardinality, parameterLearning, dataSet, statisticalTest, threshold);

            // New model is better than previous model and the difference is greater than the threshold
            if(currentModel.getScoreValue() <= bestModel.getScoreValue() && Math.abs(currentModel.getScoreValue() - bestModel.getScoreValue()) > threshold)
                return bestModel;
            else
                bestModel = currentModel;
        }

        return bestModel;
    }

    /**
     * Apply the Structural EM where the only operator is the Chow-Liu tree substitution in the TAN model
     * @return
     */
    private static LearningResult<DiscreteBayesNet> applySEM(int cardinality,
                                                             DiscreteParameterLearning parameterLearning,
                                                             DiscreteData dataSet,
                                                             DiscreteStatisticalTest statisticalTest,
                                                             double threshold){

        /** 1 - Creo un LCM de la cardinalidad indicada y le genero el CL tree */
        HLCM lcm = HlcmCreator.createLCM(dataSet.getVariables(), cardinality, new Random());
        lcm = (HLCM) parameterLearning.learnModel(lcm, dataSet).getBayesianNetwork();

        /** 2 - Estimo el Chow-Liu Tree utilizando los parametros actuales del modelo inicial */
        WeightedUndirectedGraph<DiscreteVariable> clTree = ChowLiu.learnTrueChowLiuTree(lcm.getManifestVariables(),
                lcm.getRoot().getVariable(), lcm, dataSet, statisticalTest);

        /** 3 - Creo el modelo TAN inicial */
        LearningResult<DiscreteBayesNet> initialModel = buildTAN(lcm, parameterLearning, dataSet, clTree);
        LearningResult<DiscreteBayesNet> bestModel = initialModel;
        HLCM bestModelNet = (HLCM) bestModel.getBayesianNetwork();

        /** 4 - Proceso structural EM del TAN, sigue hasta que deje de mejorar el score */
        while(true){

            /** 4.1 - Estimo el Chow-Liu Tree utilizando los parametros actuales del modelo */
            WeightedUndirectedGraph<DiscreteVariable> currentClTree = ChowLiu.learnTrueChowLiuTree(bestModelNet.getManifestVariables(),
                    lcm.getRoot().getVariable(), bestModelNet, dataSet, statisticalTest);

            /** 4.2 - Creo el model TAN */
            LearningResult<DiscreteBayesNet> currentModel = buildTAN(bestModelNet, parameterLearning, dataSet, currentClTree);

            /** 4.3 - Si el score del nuevo modelo es peor que el del "bestModel", termino el bucle y devuelvo "bestModel" */
            if(currentModel.getScoreValue() <= bestModel.getScoreValue() && Math.abs(currentModel.getScoreValue() - bestModel.getScoreValue()) > threshold)
                return bestModel;
            else
                bestModel = currentModel;
        }
    }

    private static LearningResult<DiscreteBayesNet> buildTAN(HLCM model,
                                                             DiscreteParameterLearning parameterLearning,
                                                             DiscreteData dataSet,
                                                             WeightedUndirectedGraph<DiscreteVariable> chowLiuTree){

        /** 0 - Si hay algun arco entre las variables manifest lo elimino (convirtiendolo en un LCM)*/
        List<Edge> copyEdges = new ArrayList<>(model.getEdges());
        for(Edge<Variable> edge: copyEdges){
            DiscreteBeliefNode tailNode = model.getNode(edge.getTail().getContent());
            DiscreteBeliefNode headNode = model.getNode(edge.getHead().getContent());

            if(tailNode.isManifest() && headNode.isManifest()) {
                Edge<Variable> edgeToBeRemoved = model.getEdge(headNode, tailNode).get();
                model.removeEdge(edgeToBeRemoved);
            }
        }

        /** 1 - Elijo un nodo al hacer como raiz del arbol */
        Random random = new Random();
        int rootIndex = random.nextInt(model.getManifestNodes().size() - 1);
        AbstractNode<DiscreteVariable> root = chowLiuTree.getNodes().get(rootIndex);

        /** 2 - Una vez hemos escogido la raiz del arbol, creamos un grafo dirigido al iterar recursivamente a traves del grafo */
        List<AbstractNode<DiscreteVariable>> visitedNodes = new ArrayList<>();
        visitedNodes.add(root);
        DirectedAcyclicGraph<DiscreteVariable> directedClTree = new DirectedAcyclicGraph<>();
        directedClTree.addNode(root.getContent());
        iterateChildNodes(directedClTree, visitedNodes, root);

        /** 3 - Una vez generado el DAG, añadimos sus arcos al LCM generando un TAN */
        for(Edge<DiscreteVariable> edge: directedClTree.getEdges()){
            model.addEdge(model.getNode(edge.getHead().getContent()),
                    model.getNode(edge.getTail().getContent()));
        }
        // Lo renombro para que quede claro que ha pasado a ser un TAN
        DiscreteBayesNet tan = model;

        /** 4 - The TAN parameters are learned and the model is returned */
        return parameterLearning.learnModel(tan, dataSet);
    }

    // tailRecursive method
    private static void iterateChildNodes(DirectedAcyclicGraph<DiscreteVariable> resultingGraph,
                                          List<AbstractNode<DiscreteVariable>> visitedNodes,
                                          AbstractNode<DiscreteVariable> node){

        for (AbstractNode<DiscreteVariable> neighbour: node.getNeighbors()){
            if(!visitedNodes.contains(neighbour)){

                // The neighbour node is set as visited
                visitedNodes.add(neighbour);

                // The new graph's nodes are required for adding the new edge between them
                DirectedNode<DiscreteVariable> toNeighbourNode = resultingGraph.addNode(neighbour.getContent()); // new -> added
                DirectedNode<DiscreteVariable> fromNode = resultingGraph.getNode(node.getContent()); // old -> retrieved

                resultingGraph.addEdge(toNeighbourNode, fromNode);
                iterateChildNodes(resultingGraph, visitedNodes, neighbour);
            }
        }
    }
}
