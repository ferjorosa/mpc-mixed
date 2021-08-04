package voltric.learning.structure.incremental.localemtype;

import voltric.model.DiscreteBeliefNode;
import voltric.variables.DiscreteVariable;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/* Before applying local EM we have to select which nodes are going to be updated. */
public interface TypeLocalEM {

    Set<DiscreteVariable> variablesToUpdate(List<DiscreteBeliefNode> nodes);

    default  Set<DiscreteVariable> variablesToUpdate(DiscreteBeliefNode node) {
        List<DiscreteBeliefNode> nodes = new ArrayList<>(1);
        nodes.add(node);
        return variablesToUpdate(nodes);
    }

    default Set<DiscreteVariable> variablesToUpdate(DiscreteBeliefNode firstNode, DiscreteBeliefNode secondNode){
        List<DiscreteBeliefNode> nodes = new ArrayList<>(2);
        nodes.add(firstNode);
        nodes.add(secondNode);
        return variablesToUpdate(nodes);
    }
}
