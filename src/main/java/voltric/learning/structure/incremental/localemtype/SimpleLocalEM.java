package voltric.learning.structure.incremental.localemtype;

import voltric.model.AbstractBeliefNode;
import voltric.model.DiscreteBeliefNode;
import voltric.variables.DiscreteVariable;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Before applying local EM we have to select which nodes are going to be updated. In this case we will update the
 * argument variables and their respective children.
 * */
public class SimpleLocalEM implements TypeLocalEM {

    @Override
    public Set<DiscreteVariable> variablesToUpdate(List<DiscreteBeliefNode> nodes) {

        Set<DiscreteVariable> vars = new LinkedHashSet<>();

        for(DiscreteBeliefNode node: nodes) {

            /* Add each of the variables */
            vars.add(node.getVariable());

            /* Add its children */
            for (AbstractBeliefNode child : node.getChildrenNodes())
                vars.add((DiscreteVariable) child.getVariable());

        }

        return vars;
    }
}
