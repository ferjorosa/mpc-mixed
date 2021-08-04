package voltric.learning.structure.incremental.operator;

import eu.amidst.extension.util.tuple.Tuple3;
import voltric.data.DiscreteData;
import voltric.graph.Edge;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.LocalEM;
import voltric.learning.parameter.em.config.EmConfig;
import voltric.learning.structure.incremental.localemtype.TypeLocalEM;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.variables.DiscreteVariable;
import voltric.variables.Variable;

import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;

public class LfmIncAddArc implements LfmIncOperator {

    private boolean allowObservedToObservedArc;

    private boolean allowObservedToLatentArc;

    private EmConfig emConfig;

    private TypeLocalEM typeLocalEM;

    public LfmIncAddArc(boolean allowObservedToObservedArc,
                        boolean allowObservedToLatentArc,
                        EmConfig emConfig,
                        TypeLocalEM typeLocalEM) {
        this.allowObservedToObservedArc = allowObservedToObservedArc;
        this.allowObservedToLatentArc = allowObservedToLatentArc;
        this.emConfig = emConfig;
        this.typeLocalEM = typeLocalEM;
    }

    @Override
    public Tuple3<DiscreteVariable, DiscreteVariable, LearningResult<DiscreteBayesNet>> apply(List<String> currentset,
                                                                                              DiscreteBayesNet bayesNet,
                                                                                              DiscreteData data) {

        DiscreteVariable bestFirstVar = null;
        DiscreteVariable bestSecondVar = null;
        double bestModelScore = -Double.MAX_VALUE;
        LearningResult<DiscreteBayesNet> bestModelResult = null;

        /* The BN is copied to avoid modifying current object */
        DiscreteBayesNet clonedNet = bayesNet.clone();

        /* Define the set of variables */
        List<DiscreteVariable> currentSetVariables = new ArrayList<>(currentset.size());
        for(String varName: currentset)
            currentSetVariables.add(clonedNet.getNode(varName).getVariable());

        for(DiscreteVariable firstVar: currentSetVariables) {
            for(DiscreteVariable secondVar: currentSetVariables) {
                if(!firstVar.equals(secondVar)) {
                    if(firstVar.isLatentVariable() && secondVar.isManifestVariable()                                                // LV -> OV
                            || firstVar.isLatentVariable() && secondVar.isLatentVariable()                                          // LV -> LV
                            || firstVar.isManifestVariable() && secondVar.isManifestVariable() && this.allowObservedToObservedArc   // OV -> OV
                            || firstVar.isManifestVariable() && secondVar.isLatentVariable() && this.allowObservedToLatentArc) {    // OV -> LV

                        DiscreteBeliefNode firstNode = clonedNet.getNode(firstVar);
                        DiscreteBeliefNode secondNode = clonedNet.getNode(secondVar);

                        /* Create an arc from the firstVar to the secondVar */
                        // NOTE: In Voltric the edge direction is inverse to AMIDST
                        Edge<Variable> edge = clonedNet.addEdge(secondNode, firstNode);

                        /* Locally learn the parameters of the new model (consider the markov blankets of the edge nodes) */
                        Set<DiscreteVariable> localVariables = typeLocalEM.variablesToUpdate(firstNode, secondNode);
                        LocalEM localEM = new LocalEM(localVariables, emConfig);
                        LearningResult<DiscreteBayesNet> newEdgeResult = localEM.learnModel(clonedNet, data);

                        /* Store the new model if score improves */
                        if (newEdgeResult.getScoreValue() > bestModelScore) {
                            bestModelScore = newEdgeResult.getScoreValue();
                            bestModelResult = newEdgeResult;
                            bestFirstVar = firstVar;
                            bestSecondVar = secondVar;
                        }

                        /* Independently of the score, remove the new edge*/
                        clonedNet.removeEdge(edge);
                    }
                }
            }
        }

        if(bestModelResult != null) {
            bestModelResult.setName("AddArc");
            return new Tuple3<>(bestFirstVar, bestSecondVar, bestModelResult);
        }

        /* This case implies that no model have improved the -Double.MAX_VALUE score */
        return new Tuple3<>(bestFirstVar, bestSecondVar, new LearningResult<>(bayesNet, bestModelScore, emConfig.getScoreType(), "AddArc"));
    }

    @Override
    public Tuple3<DiscreteVariable, DiscreteVariable, LearningResult<DiscreteBayesNet>> apply(PriorityQueue<Tuple3<String, String, Double>> selectedTriples,
                                                                                              DiscreteBayesNet bayesNet,
                                                                                              DiscreteData data) {

        DiscreteVariable bestFirstVar = null;
        DiscreteVariable bestSecondVar = null;
        double bestModelScore = -Double.MAX_VALUE;
        LearningResult<DiscreteBayesNet> bestModelResult = null;

        /* The BN is copied to avoid modifying current object */
        DiscreteBayesNet clonedNet = bayesNet.clone();

        /* Iterate through the selected triples */
        for(Tuple3<String, String, Double> triple: selectedTriples){

            /* Create a tuple with the variables' copies as items */
            List<DiscreteVariable> tupleList = new ArrayList<>(2);
            tupleList.add(clonedNet.getNode(triple.getFirst()).getVariable());
            tupleList.add(clonedNet.getNode(triple.getSecond()).getVariable());

            for(DiscreteVariable firstVar: tupleList) {
                for (DiscreteVariable secondVar : tupleList) {
                    if (!firstVar.equals(secondVar)) {
                        if (firstVar.isLatentVariable() && secondVar.isManifestVariable()                                               // LV -> OV
                                || firstVar.isLatentVariable() && secondVar.isLatentVariable()                                          // LV -> LV
                                || firstVar.isManifestVariable() && secondVar.isManifestVariable() && this.allowObservedToObservedArc   // OV -> OV
                                || firstVar.isManifestVariable() && secondVar.isLatentVariable() && this.allowObservedToLatentArc) {    // OV -> LV

                            DiscreteBeliefNode firstNode = clonedNet.getNode(firstVar);
                            DiscreteBeliefNode secondNode = clonedNet.getNode(secondVar);

                            /* Create an arc from the firstVar to the secondVar */
                            // NOTE: In Voltric the edge direction is inverse to AMIDST
                            Edge<Variable> edge = clonedNet.addEdge(secondNode, firstNode);

                            /* Locally learn the parameters of the new model (consider the markov blankets of the edge nodes) */
                            Set<DiscreteVariable> localVariables = typeLocalEM.variablesToUpdate(firstNode, secondNode);
                            LocalEM localEM = new LocalEM(localVariables, emConfig);
                            LearningResult<DiscreteBayesNet> newEdgeResult = localEM.learnModel(clonedNet, data);

                            /* Store the new model if score improves */
                            if (newEdgeResult.getScoreValue() > bestModelScore) {
                                bestModelScore = newEdgeResult.getScoreValue();
                                bestModelResult = newEdgeResult;
                                bestFirstVar = firstVar;
                                bestSecondVar = secondVar;
                            }

                            /* Independently of the score, remove the new edge*/
                            clonedNet.removeEdge(edge);
                        }
                    }
                }
            }
        }

        if(bestModelResult != null) {
            bestModelResult.setName("AddArc");
            return new Tuple3<>(bestFirstVar, bestSecondVar, bestModelResult);
        }

        /* This case implies that no model have improved the -Double.MAX_VALUE score */
        return new Tuple3<>(bestFirstVar, bestSecondVar, new LearningResult<>(bayesNet, bestModelScore, emConfig.getScoreType(), "AddArc"));
    }
}
