package voltric.learning.structure.incremental.localemtype;

import voltric.model.AbstractBeliefNode;
import voltric.model.DiscreteBeliefNode;
import voltric.variables.DiscreteVariable;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Before applying local EM we have to select which nodes are going to be updated. In this case we will update the
 * argument variables and their combined (overlapping) markov blanket
 * */
public class MarkovBlanketLocalEM implements TypeLocalEM {

    @Override
    public Set<DiscreteVariable> variablesToUpdate(List<DiscreteBeliefNode> nodes) {
        Set<DiscreteVariable> overlappingMarkovBlanketVars = new LinkedHashSet<>();

        for(DiscreteBeliefNode node: nodes) {

            //overlappingMarkovBlanketVars.add(node.getVariable());

            /* Parents */
            for (AbstractBeliefNode parent : node.getParentNodes())
                overlappingMarkovBlanketVars.add((DiscreteVariable) parent.getVariable());

            /* Children and their parents */
            for (AbstractBeliefNode child : node.getChildrenNodes()) {
                overlappingMarkovBlanketVars.add((DiscreteVariable) child.getVariable());
                for (AbstractBeliefNode childParent : child.getParentNodes())
                    overlappingMarkovBlanketVars.add((DiscreteVariable) childParent.getVariable());
            }
        }

        return overlappingMarkovBlanketVars;
    }
}
