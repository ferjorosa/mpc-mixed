package voltric.learning.structure.hillclimbing.local;

import voltric.data.DiscreteData;
import voltric.graph.Edge;
import voltric.learning.score.ScoreType;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.variables.DiscreteVariable;
import voltric.variables.Variable;

import java.util.*;

// TODO: Una vez ha buscado todos los nodos que no se encuentran en las blacklists, se queda con aquellos que mejoran el score y filtra los que no cumplan con la estructura
public class LocalDeleteArc implements LocalHcOperator {

    /** The set of nodes that need to be avoided in the structure search process. All the edges containing black-listed
     * nodes will be avoided.
     */
    private List<DiscreteVariable> blackList;

    /**
     * The set of edges that need to be avoided in the structure search process. The key of the map is the tail node
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
    public LocalDeleteArc(List<DiscreteVariable> blackList, Map<DiscreteVariable, List<DiscreteVariable>> edgeBlackList){
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
    public LocalDeleteArc(List<DiscreteVariable> blackList, List<Edge<DiscreteVariable>> edgeBlackList){
        this.blackList = blackList;
        this.edgeBlackList = new HashMap<>();

        for(Edge<DiscreteVariable> edge: edgeBlackList)
            this.edgeBlackList.put(edge.getTail().getContent(), new ArrayList<>());

        for(Edge<DiscreteVariable> edge: edgeBlackList)
            this.edgeBlackList.get(edge.getTail().getContent()).add(edge.getHead().getContent());
    }


    @Override
    public List<LocalOperation> apply(DiscreteBayesNet seedNet, DiscreteData data, EfficientDiscreteData efficientData, Map<DiscreteVariable, Double> scores, ScoreType scoreType) {

        List<LocalOperation> operations = new LinkedList<>(); // We use a linkedList because it is NOT going to be randomly accessed

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
                double deleteArcScore = LocalScore.computeLocalScore(edgeHeadVar, newFamily, data, efficientData, scoreType);

                double localScore = 0;
                localScore += deleteArcScore;

                for(DiscreteVariable variable: scores.keySet())
                    if(!variable.equals(edgeHeadVar))
                        localScore += scores.get(variable);

                /** Add the operation to the list */
                operations.add(new LocalOperation(edgeHeadVar, edgeTailVar, localScore, LocalOperation.Type.OPERATION_DEL));
            }
        }

        return operations;
    }

    void copyArcRestrictions(DiscreteVariable existingVariable, DiscreteVariable newVariable) {

        // Caso en que no se pueden añadir arcos DESDE ELLA
        if(this.edgeBlackList.containsKey(existingVariable)) {
            List<DiscreteVariable> toVariableList = this.edgeBlackList.get(existingVariable);
            this.edgeBlackList.put(newVariable, new ArrayList<>(toVariableList));
        }

        // Caso en el que no se pueden añadir arcos A ELLA
        for(DiscreteVariable fromVariable: this.edgeBlackList.keySet()){
            List<DiscreteVariable> toVariableList = this.edgeBlackList.get(fromVariable);
            if(toVariableList.contains(existingVariable))
                toVariableList.add(newVariable);
        }
    }
}
