package voltric.learning.structure.hillclimbing.efficientlocal;

import voltric.data.DiscreteData;
import voltric.graph.Edge;
import voltric.learning.score.ScoreType;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.variables.DiscreteVariable;
import voltric.variables.Variable;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

// TODO: Podria combinar en calculateNodeScore el tema de estimar la nueva CPT con los datos proyectados
public class EffLocalReverseArc implements EffLocalHcOperator {

    /** The set of nodes that need to be avoided in the structure search process. */
    private List<DiscreteVariable> blackList;

    /** The set of edges that need to be avoided in the structure search process. */
    private Map<DiscreteVariable, List<DiscreteVariable>> edgeBlackList;

    /** Maximum number of parents nodes. */
    private int maxNumberOfParents;

    /**
     * Main constructor.
     *
     * @param blackList The set of nodes that need to be avoided in the structure search process. All the edges that contain
     *                  a black-listed node will be avoided.
     * @param edgeBlackList the set of edges that need to be avoided in the structure search process. The key of the map
     *                      is the tail node and the List contains all the head nodes.
     * @param maxNumberOfParents The maximum number of parent nodes.
     */
    public EffLocalReverseArc(List<DiscreteVariable> blackList, Map<DiscreteVariable, List<DiscreteVariable>> edgeBlackList,  int maxNumberOfParents){
        this.blackList = blackList;
        this.edgeBlackList = edgeBlackList;
        this.maxNumberOfParents = maxNumberOfParents;
    }

    /**
     * This constructor accepts a Set of edges (to avoid repeated ones) as the black list.
     *
     * @param blackList The set of nodes that need to be avoided in the structure search process. All the edges that contain
     *                  a black-listed node will be avoided.
     * @param edgeBlackList The set of edges that need to be avoided in the structure search process.
     * @param maxNumberOfParents The maximum number of parent nodes.
     */
    public EffLocalReverseArc(List<DiscreteVariable> blackList, List<Edge<DiscreteVariable>> edgeBlackList, int maxNumberOfParents){
        this.blackList = blackList;
        this.edgeBlackList = new HashMap<>();
        this.maxNumberOfParents = maxNumberOfParents;

        for(Edge<DiscreteVariable> edge: edgeBlackList)
            this.edgeBlackList.put(edge.getTail().getContent(), new ArrayList<>());

        for(Edge<DiscreteVariable> edge: edgeBlackList)
            this.edgeBlackList.get(edge.getTail().getContent()).add(edge.getHead().getContent());
    }

    @Override
    public EffLocalOperation apply(DiscreteBayesNet seedNet, DiscreteData data, EfficientDiscreteData efficientData, Map<DiscreteVariable, Double> scores, ScoreType scoreType) {

        DiscreteVariable bestEdgeHead = null;
        DiscreteVariable bestEdgeTail = null;
        double bestEdgeScore = -Double.MAX_VALUE; // Log-likelihood related scores are negative

        /** Iteration through all the edges in the BN */
        for (Edge<Variable> edge : seedNet.getEdges()) {

            DiscreteBeliefNode edgeTailNode = seedNet.getNode(edge.getTail().getContent());
            DiscreteVariable edgeTailVar = edgeTailNode.getVariable();

            DiscreteBeliefNode edgeHeadNode = seedNet.getNode(edge.getHead().getContent());
            DiscreteVariable edgeHeadVar = edgeHeadNode.getVariable();

            /** Check that none of the edge's nodes is black-listed */
            if (!blackList.contains(edgeTailVar) && !blackList.contains(edgeHeadVar) &&
                    /** Check that the edge to be reversed is not forbidden */
                    (!edgeBlackList.containsKey(edgeTailVar) || !edgeBlackList.get(edgeTailVar).contains(edgeHeadVar)) &&
                    /** Check that the reversed edge is allowed (from headNode to tailNode) */
                    seedNet.isEdgeAllowed(edgeTailNode, edgeHeadNode) &&
                    /** Check that the reversed edge wouldn't surpass the allowed number of parent nodes */
                    seedNet.getNode(edgeTailVar).getParentNodes().size() < this.maxNumberOfParents) {

                List<DiscreteVariable> newTailFamily = edgeTailNode.getDiscreteParentVariables();
                newTailFamily.add(edgeTailVar);
                newTailFamily.add(edgeHeadVar); // Add the edgeHead as another parent var in the family

                List<DiscreteVariable> newHeadFamily = edgeHeadNode.getDiscreteParentVariables();
                newHeadFamily.add(edgeHeadVar);
                newHeadFamily.remove(edgeTailVar); // Remove the edgeTail node from the family

                /** Calculate the local score corresponding to the arc being reversed and update the bestScore if necessary */
                double newTailScore = EffLocalScore.computeLocalScore(edgeTailVar, newTailFamily, data, efficientData, scoreType);
                double newHeadScore = EffLocalScore.computeLocalScore(edgeHeadVar, newHeadFamily, data, efficientData, scoreType);

                double localScore = 0;
                localScore += newTailScore;
                localScore += newHeadScore;

                for (DiscreteVariable variable : scores.keySet())
                    if (!variable.equals(edgeTailVar) && !variable.equals(edgeHeadVar))
                        localScore += scores.get(variable);

                if (localScore > bestEdgeScore) {
                    bestEdgeScore = localScore;
                    bestEdgeHead = edgeHeadVar;
                    bestEdgeTail = edgeTailVar;
                }
            }
        }

        return new EffLocalOperation(bestEdgeHead, bestEdgeTail, bestEdgeScore, EffLocalOperation.Type.OPERATION_REVERSE);
    }
}
