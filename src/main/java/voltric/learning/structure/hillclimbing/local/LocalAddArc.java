package voltric.learning.structure.hillclimbing.local;

import voltric.data.DiscreteData;
import voltric.learning.score.ScoreType;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.variables.DiscreteVariable;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

// TODO: Una vez ha buscado todos los nodos que no se encuentran en las blacklists, se queda con aquellos que mejoran el score y filtra los que no cumplan con la estructura
public class LocalAddArc implements LocalHcOperator {

    /** The set of nodes that need to be avoided in the structure search process. */
    private List<DiscreteVariable> blackList;

    /**
     * The set of edges that need to be avoided in the structure search process. The key of the map is the tail node
     * and the List contains all the head nodes.
     */
    // In MBC: Feature variable, List of class variables
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
    public LocalAddArc(List<DiscreteVariable> blackList, Map<DiscreteVariable, List<DiscreteVariable>> edgeBlackList, int maxNumberOfParents){
        this.blackList = blackList;
        this.edgeBlackList = edgeBlackList;
        this.maxNumberOfParents = maxNumberOfParents;
    }

    @Override
    public List<LocalOperation> apply(DiscreteBayesNet seedNet, DiscreteData data, EfficientDiscreteData efficientData, Map<DiscreteVariable, Double> scores, ScoreType scoreType) {

        /** The BN nodes are filtered using the blacklist. */
        List<DiscreteBeliefNode> whiteList = seedNet.getVariables().stream()
                .filter(x -> !this.blackList.contains(x))
                .map(var -> seedNet.getNode(var))
                .collect(Collectors.toList());

        List<LocalOperation> operations = new LinkedList<>(); // We use a linkedList because it is NOT going to be randomly accessed

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
                    double addArcScore = LocalScore.computeLocalScore(toNode.getVariable(), newFamily, data, efficientData, scoreType);

                    double localScore = 0;
                    localScore += addArcScore;

                    for(DiscreteVariable variable: scores.keySet())
                        if(!variable.equals(toNode.getVariable()))
                            localScore += scores.get(variable);

                    /** Add the operation to the list */
                    operations.add(new LocalOperation(toNode.getVariable(), fromNode.getVariable(), localScore, LocalOperation.Type.OPERATION_ADD));

                } // end-if
            }  // end-for
        } // end-for

        return operations;
    }

    void addEdgeToBlackList(DiscreteVariable fromVariable, DiscreteVariable toVariable){
        if(!edgeBlackList.containsKey(fromVariable))
            edgeBlackList.put(fromVariable, new ArrayList<>());

        List<DiscreteVariable> toVariableList = this.edgeBlackList.get(fromVariable);
        toVariableList.add(toVariable);
    }

    // Itera por el conjunto de keys y elimina la variable en cuestion de la lista asociada a la Key
    void removeNodeFromEdgeBlackList(DiscreteVariable variable) {

        // Caso para los arcos DESDE ELLA
        this.edgeBlackList.remove(variable); // fromVariable

        // Caso para los arcos A ELLA
        for(DiscreteVariable fromVariable: edgeBlackList.keySet()) {
            List<DiscreteVariable> toVariableList = edgeBlackList.get(fromVariable);
            toVariableList.remove(variable); // toVariable
        }
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
