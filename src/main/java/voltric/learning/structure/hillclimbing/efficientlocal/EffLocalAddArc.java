package voltric.learning.structure.hillclimbing.efficientlocal;

import voltric.data.DiscreteData;
import voltric.learning.score.ScoreType;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.variables.DiscreteVariable;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

// TODO: Una vez ha buscado todos los nodos que no se encuentran en las blacklists, se queda con aquellos que mejoran el score y filtra los que no cumplan con la estructura
public class EffLocalAddArc implements EffLocalHcOperator {

    /** The set of nodes that need to be avoided in the structure search process. */
    private List<DiscreteVariable> blackList;

    /** The set of edges that need to be avoided in the structure search process. */
    private Map<DiscreteVariable, List<DiscreteVariable>> edgeBlackList;


    /** Maximum number of parent nodes. */
    private int maxNumberOfParents;

    /**
     * Main constructor.
     *
     * @param blackList The set of nodes that need to be avoided in the structure search process. All the edges that contain
     *                  a black-listed node will be avoided.
     * @param edgeBlackList The set of edges that need to be avoided in the structure search process. The key of the map
     *                      is the tail node and the List contains all the head nodes.
     */
    public EffLocalAddArc(List<DiscreteVariable> blackList, Map<DiscreteVariable, List<DiscreteVariable>> edgeBlackList, int maxNumberOfParents){
        this.blackList = blackList;
        this.edgeBlackList = edgeBlackList;
        this.maxNumberOfParents = maxNumberOfParents;
    }

    @Override
    public EffLocalOperation apply(DiscreteBayesNet seedNet, DiscreteData data, EfficientDiscreteData efficientData, Map<DiscreteVariable, Double> scores, ScoreType scoreType) {

        /** The BN nodes are filtered using the blacklist. */
        List<DiscreteBeliefNode> whiteList = seedNet.getVariables().stream()
                .filter(x -> !this.blackList.contains(x))
                .map(var -> seedNet.getNode(var))
                .collect(Collectors.toList());

        DiscreteVariable bestEdgeHead = null;
        DiscreteVariable bestEdgeTail = null;
        double bestEdgeScore = -Double.MAX_VALUE; // Log-likelihood related scores are negative

        /** Iteration through all the white-listed BN nodes */
        for(DiscreteBeliefNode fromNode : whiteList) {
            for (DiscreteBeliefNode toNode : whiteList) {

                /** Check the arc to be added is not forbidden */
                if ((!edgeBlackList.containsKey(fromNode.getVariable()) || !edgeBlackList.get(fromNode.getVariable()).contains(toNode.getVariable())) &&
                        /** Then it checks if the edge is allowed */
                        seedNet.isEdgeAllowed(toNode, fromNode) &&
                        /** Then it checks this new edge wouldn't surpass the allowed number of parent edges of the receiving node */
                        toNode.getParentNodes().size() < this.maxNumberOfParents) {

                    /** Project data using the node's new family */
                    List<DiscreteVariable> newFamily = toNode.getDiscreteParentVariables();
                    newFamily.add(toNode.getVariable());
                    newFamily.add(fromNode.getVariable());

                    /** Calculate the local score corresponding to the new arc and update the bestScore if necessary */
                    double addArcScore = EffLocalScore.computeLocalScore(toNode.getVariable(), newFamily, data, efficientData, scoreType);

                    double localScore = 0;
                    localScore += addArcScore;

                    for(DiscreteVariable variable: scores.keySet())
                        if(!variable.equals(toNode.getVariable()))
                            localScore += scores.get(variable);

                    if (localScore > bestEdgeScore) {
                        bestEdgeScore = localScore;
                        bestEdgeHead = toNode.getVariable();
                        bestEdgeTail = fromNode.getVariable();
                    }

                } // end-if
            }  // end-for
        } // end-for

        return new EffLocalOperation(bestEdgeHead, bestEdgeTail, bestEdgeScore, EffLocalOperation.Type.OPERATION_ADD);
    }
}
