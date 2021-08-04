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

// TODO: Una vez ha buscado todos los nodos que no se encuentran en las blacklists, se queda con aquellos que mejoran el score y filtra los que no cumplan con la estructura
public class EffLocalDeleteArc implements EffLocalHcOperator {

    /** The set of nodes that need to be avoided in the structure search process. All the edges containing black-listed
     * nodes will be avoided.
     */
    private List<DiscreteVariable> blackList;

    /** The set of edges that need to be avoided in the structure search process. The key of the map is the tail node
     * and the List contains all the head nodes.
     */
    private Map<DiscreteVariable, List<DiscreteVariable>> edgeBlackList;

    /**
     * Main constructor.
     *
     * @param blackList The set of nodes that need to be avoided in the structure search process. All the edges that contain
     *                  a black-listed node will be avoided.
     * @param edgeBlackList the set of edges that need to be avoided in the structure search process. The key of the map
     *                      is the tail node and the List contains all the head nodes.
     */
    public EffLocalDeleteArc(List<DiscreteVariable> blackList, Map<DiscreteVariable, List<DiscreteVariable>> edgeBlackList){
        this.blackList = blackList;
        this.edgeBlackList = edgeBlackList;
    }

    /**
     * This constructor accepts a Set of edges (to avoid repeated ones) as the black list.
     *
     * @param blackList The set of nodes that need to be avoided in the structure search process. All the edges that contain
     *                  a black-listed node will be avoided.
     * @param edgeBlackList The set of edges that need to be avoided in the structure search process.
     */
    public EffLocalDeleteArc(List<DiscreteVariable> blackList, List<Edge<DiscreteVariable>> edgeBlackList){
        this.blackList = blackList;
        this.edgeBlackList = new HashMap<>();

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
        for(Edge<Variable> edge: seedNet.getEdges()) {

            DiscreteBeliefNode edgeTailNode = seedNet.getNode(edge.getTail().getContent());
            DiscreteVariable edgeTailVar = edgeTailNode.getVariable();

            DiscreteBeliefNode edgeHeadNode = seedNet.getNode(edge.getHead().getContent());
            DiscreteVariable edgeHeadVar = edgeHeadNode.getVariable();

            /** Check that none of the edge's nodes is black-listed */
            if (!blackList.contains(edgeTailVar) && !blackList.contains(edgeHeadVar) &&
                    /** Check that the edge to be removed is not forbidden */
                    (!edgeBlackList.containsKey(edgeTailVar) || !edgeBlackList.get(edgeTailVar).contains(edgeHeadVar))) {

                /** Project data using the node's new family */
                List<DiscreteVariable> newFamily = edgeHeadNode.getDiscreteParentVariables();
                newFamily.add(edgeHeadVar);
                newFamily.remove(edgeTailVar); // Remove the edgeTail node from the family

                /** Calculate the local score corresponding to the arc being removed and update the bestScore if necessary */
                double deleteArcScore = EffLocalScore.computeLocalScore(edgeHeadVar, newFamily, data, efficientData, scoreType);

                double localScore = 0;
                localScore += deleteArcScore;

                for(DiscreteVariable variable: scores.keySet())
                    if(!variable.equals(edgeHeadVar))
                        localScore += scores.get(variable);

                if (localScore > bestEdgeScore) {
                    bestEdgeScore = localScore;
                    bestEdgeHead = edgeHeadVar;
                    bestEdgeTail = edgeTailVar;
                }
            }
        }

        return new EffLocalOperation(bestEdgeHead, bestEdgeTail, bestEdgeScore, EffLocalOperation.Type.OPERATION_DEL);
    }
}
