package voltric.learning.structure.type;

import voltric.graph.DirectedAcyclicGraph;
import voltric.model.AbstractBayesNet;
import voltric.variables.Variable;

/**
 * Created by fernando on 22/08/17.
 */
public class TreeStructure implements StructureType {

    @Override
    public boolean allows(AbstractBayesNet bayesNet) {
        return bayesNet.isTree();
    }

    @Override
    public boolean allows(DirectedAcyclicGraph<Variable> dag) {
        return dag.isTree();
    }
}
