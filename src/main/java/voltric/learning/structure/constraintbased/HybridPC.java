package voltric.learning.structure.constraintbased;

import voltric.data.DiscreteData;
import voltric.graph.AbstractNode;
import voltric.graph.UndirectedGraph;
import voltric.learning.LearningResult;
import voltric.learning.parameter.DiscreteParameterLearning;
import voltric.learning.structure.hillclimbing.global.*;
import voltric.learning.structure.type.DagStructure;
import voltric.model.DiscreteBayesNet;
import voltric.variables.DiscreteVariable;
import voltric.variables.Variable;

import java.util.*;

/**
 * The PC algorithm returns a PDAG class, so to be able to return a single BN model, we create this hybrid version that
 * applies a Hill-climbing algorithm to the returned PDAG object with the objective of generating a DAG model.
 *
 * TODO: Recordar que hay otras posibilidades como la propuesta por Chickering en la pag 454 (ultimo parrafo) de su articulo http://www.jmlr.org/papers/volume2/chickering02a/chickering02a.pdf
 * el cual se aplciaria tras el segundo paso del PC una vez generado un PDAG.
 */
//TODO: Esta implementación se hace siguiendo con un test de independencia de normalized MI y chi-square
public class HybridPC {

    private CBLearner cbLearner;

    private int maxIterationsHC;

    private double threshold;

    UndirectedGraph<DiscreteVariable> skeleton;

    public HybridPC(CBLearner cbLearner, int maxIterationsHC, double threshold){
        this.cbLearner = cbLearner;
        this.maxIterationsHC = maxIterationsHC;
        this.threshold = threshold;
    }

    public UndirectedGraph<DiscreteVariable> getSkeleton(){ return this.skeleton; }

    // TODO: Vamos a empezar por el caso mas simple, y luego pasaremos a la versión concurrente
    public LearningResult<DiscreteBayesNet> learnModel(DiscreteData data, DiscreteParameterLearning parameterLearning) {

        /** First the CB learner is called to produce a skeleton of the Bayesian network */
        this.skeleton = this.cbLearner.learnSkeleton(data);

        /** Then we initiate the Hill-climbing algorithm that is going to be used */

        // First we create the initial model for the HC, which is going to be a model with no edges
        DiscreteBayesNet initialModel = new DiscreteBayesNet();
        for(DiscreteVariable variable: data.getVariables())
            initialModel.addNode(variable);

        // Then we create the operators that are going to take action

        Map<Variable, List<Variable>> edgeBlackList = new HashMap<>();
        for(AbstractNode<DiscreteVariable> node: skeleton.getNodes())
            edgeBlackList.put(node.getContent(), new ArrayList<>());

        // For each pair of nodes, if none of them have a connection in the skeleton, an edge is created and added to the black list
        for(AbstractNode<DiscreteVariable> node: skeleton.getNodes())
            for(AbstractNode<DiscreteVariable> neighbor: skeleton.getNodes()){
                if(!node.equals(neighbor) && !skeleton.containsEdge(neighbor, node))
                    edgeBlackList
                            .get(node.getContent()) // Returns the list of black-listed neighbors
                            .add(neighbor.getContent()); // Adds a new one
            }

        AddArc addArcOperator = new AddArc(new ArrayList<>(), edgeBlackList, new DagStructure(), cbLearner.getMaxParentOrder());
        // The remove and reverse operators have a set of forbidden edges, those that go from the root to the manifest nodes.
        RemoveArc removeArcOperator = new RemoveArc(new ArrayList<>(), new ArrayList<>(), new DagStructure());
        ReverseArc reverseArcOperator = new ReverseArc(new ArrayList<>(), new ArrayList<>(), new DagStructure());

        Set<HcOperator> operatorSet = new HashSet<>();
        operatorSet.add(addArcOperator);
        //operatorSet.add(removeArcOperator);
        //operatorSet.add(reverseArcOperator);

        // The number of iterations isn't taken into consideration, that is why a fixed number like 10 is assigned
        GlobalHillClimbing hillClimbing = new GlobalHillClimbing(operatorSet, this.maxIterationsHC, this.threshold);

        /** The hill-climbing process returns a solution */
        return hillClimbing.learnModel(initialModel, data, parameterLearning);
    }
}
