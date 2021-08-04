package voltric.learning.structure.type;

import voltric.graph.DirectedAcyclicGraph;
import voltric.model.AbstractBayesNet;
import voltric.variables.Variable;

public class ForestStructure implements StructureType {

    @Override
    public boolean allows(AbstractBayesNet bayesNet) {
        return bayesNet.isForest();
    }

    @Override
    public boolean allows(DirectedAcyclicGraph<Variable> dag) {
        return dag.isForest();
    }
}
