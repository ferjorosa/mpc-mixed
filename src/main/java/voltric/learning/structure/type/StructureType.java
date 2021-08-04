package voltric.learning.structure.type;

import voltric.graph.DirectedAcyclicGraph;
import voltric.model.AbstractBayesNet;
import voltric.variables.Variable;

/**
 * Other possibility would be PolyTrees.
 */
public interface StructureType {

    boolean allows(AbstractBayesNet bayesNet);

    boolean allows(DirectedAcyclicGraph<Variable> dag);
}
