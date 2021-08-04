package voltric.clustering.unidimensional;

import voltric.data.DiscreteData;
import voltric.graph.Edge;
import voltric.learning.LearningResult;
import voltric.learning.parameter.DiscreteParameterLearning;
import voltric.learning.structure.hillclimbing.global.*;
import voltric.learning.structure.type.DagStructure;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.model.HLCM;
import voltric.model.creator.HlcmCreator;
import voltric.variables.DiscreteVariable;
import voltric.variables.Variable;
import voltric.variables.modelTypes.VariableType;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by fernando on 25/08/17.
 *
 * Tengo en ejemplos ya hecha la implementacion con Structural EM (proyecto new-parkinson), esta clase no tiene sentido, hay que actualizarla
 */
@Deprecated
public class OldLearnLatentKDB {

    // K-DB variant that allows an initial network that connects the MVs of the k-db.
    public static LearningResult<DiscreteBayesNet> learnModel(DiscreteBayesNet seedtNet,
                                                              int maxCardinality,
                                                              DiscreteData dataSet,
                                                              DiscreteParameterLearning parameterLearning,
                                                              double threshold,
                                                              int maxParents,
                                                              int maxIterations){

        /** First a K-db model of cardinality 2 is created based on the seedNet's network */
        DiscreteBayesNet initialModel = new DiscreteBayesNet();

        // The root variable is created
        DiscreteBeliefNode root = initialModel.addNode(new DiscreteVariable(2, VariableType.LATENT_VARIABLE));

        // All the manifest variables of the seedNet are added to the new BN and an edge from the root to them is created
        for(DiscreteVariable variable: seedtNet.getManifestVariables())
            initialModel.addEdge(initialModel.addNode(variable), root);

        // Al the edges of the seedNet are also added to the initial model
        for(Edge<Variable> edge: seedtNet.getEdges()){
            Variable edgeTail = edge.getTail().getContent();
            Variable edgeHead = edge.getHead().getContent();
            initialModel.addEdge(initialModel.getNode(edgeHead), initialModel.getNode(edgeTail));
        }

        // Finally the model is randomly parametrized
        initialModel.randomlyParameterize();

        /** A hill-climbing search process is applied where 5 operators are allowed */
        IncreaseLatentCardinality ilcOperator = new IncreaseLatentCardinality(maxCardinality);
        DecreaseLatentCardinality dlcOperator = new DecreaseLatentCardinality();
        AddArc addArcOperator = new AddArc(new DagStructure(), maxParents);

        // The remove and reverse operators have a set of forbidden edges, those that go from the root to the manifest nodes.
        List<Edge<Variable>> forbiddenEdges = initialModel.getEdges().stream().filter(x-> x.getTail().equals(root)).collect(Collectors.toList());

        RemoveArc removeArcOperator = new RemoveArc(new ArrayList<>(), forbiddenEdges, new DagStructure());
        ReverseArc reverseArcOperator = new ReverseArc(new ArrayList<>(), forbiddenEdges, new DagStructure());

        Set<HcOperator> operatorSet = new HashSet<>();
        operatorSet.add(ilcOperator);
        operatorSet.add(dlcOperator);
        operatorSet.add(addArcOperator);
        operatorSet.add(removeArcOperator);
        operatorSet.add(reverseArcOperator);

        GlobalHillClimbing hillClimbing = new GlobalHillClimbing(operatorSet, maxIterations, threshold);

        return hillClimbing.learnModel(initialModel, dataSet, parameterLearning);
    }

    // Simple KDB variant where only adding edges is possible or increasing / decreasing the cardinality
    public static LearningResult<DiscreteBayesNet> learnSimpleModel(int maxCardinality,
                                                              DiscreteData dataSet,
                                                              DiscreteParameterLearning parameterLearning,
                                                              double threshold,
                                                              int maxParents,
                                                              int maxIterations){

        // First the initial model is created
        HLCM initialModel = HlcmCreator.createLCM(dataSet.getVariables(), 2, new Random());

        // A hill-climbing search process is applied where 5 operators are allowed
        IncreaseLatentCardinality ilcOperator = new IncreaseLatentCardinality(maxCardinality);
        DecreaseLatentCardinality dlcOperator = new DecreaseLatentCardinality();
        AddArc addArcOperator = new AddArc(new DagStructure(), maxParents);
        // The remove and reverse operators have a set of forbidden edges, those that go from the root to the manifest nodes.
        List<Edge<Variable>> forbiddenEdges = initialModel.getEdges().stream().filter(x-> x.getTail().equals(initialModel.getRoot())).collect(Collectors.toList());

        RemoveArc removeArcOperator = new RemoveArc(new ArrayList<>(), forbiddenEdges, new DagStructure());
        ReverseArc reverseArcOperator = new ReverseArc(new ArrayList<>(), forbiddenEdges, new DagStructure());

        Set<HcOperator> operatorSet = new HashSet<>();
        operatorSet.add(ilcOperator);
        operatorSet.add(dlcOperator);
        operatorSet.add(addArcOperator);
        //operatorSet.add(removeArcOperator);
        //operatorSet.add(reverseArcOperator);

        GlobalHillClimbing hillClimbing = new GlobalHillClimbing(operatorSet, maxIterations, threshold);

        return hillClimbing.learnModel(initialModel, dataSet, parameterLearning);
    }
}
