package voltric.learning.structure.hillclimbing.global;

import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.parameter.DiscreteParameterLearning;
import voltric.model.DiscreteBayesNet;
import voltric.variables.DiscreteVariable;
import voltric.variables.Variable;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by equipo on 02/08/2017.
 */
public class DecreaseLatentCardinality implements HcOperator {

    /** The set of nodes that need to be avoided in the structure search process. */
    private List<Variable> blackList;

    private int minCardinality;

    /**
     * Main constructor.
     *
     * @param blackList The set of nodes that need to be avoided in the structure search process.
     */
    public DecreaseLatentCardinality(List<Variable> blackList, int minCardinality){

        if(minCardinality < 2)
            throw  new IllegalArgumentException("The minimum cardinality value for a categorical variable is 2");

        this.blackList = blackList;
        this.minCardinality = minCardinality;
    }

    public DecreaseLatentCardinality(int minCardinality){
        this(new ArrayList<>(), minCardinality);
    }

    public DecreaseLatentCardinality(){
        this(new ArrayList<>(), 2);
    }

    /** {@inheritDoc} */
    @Override
    public LearningResult<DiscreteBayesNet> apply(DiscreteBayesNet seedNet, DiscreteData data, DiscreteParameterLearning parameterLearning) {
        // The BN is copied to avoid modifying current object.
        DiscreteBayesNet clonedNet = seedNet.clone();

        // The BN latent nodes are filtered using the blacklist
        List<DiscreteVariable> whiteList = clonedNet.getLatentVariables().stream()
                .filter(x -> !this.blackList.contains(x))
                .collect(Collectors.toList());

        LearningResult<DiscreteBayesNet> bestModel = new LearningResult<>(null, -Double.MAX_VALUE, parameterLearning.getScoreType());

        // Iteration through all the allowed BN nodes
        for(DiscreteVariable latentVar : whiteList) {

            // The cardinality of the LV must be lesser than the established maximum
            if(latentVar.getCardinality() > this.minCardinality) {

                clonedNet = clonedNet.decreaseCardinality(clonedNet.getLatentVariable(latentVar.getName()), 1);

                // After the LV has increased its cardinality, the resulting model is learned. If its score is improved, the LV is stored
                LearningResult<DiscreteBayesNet> newModel = parameterLearning.learnModel(clonedNet, data);
                if (newModel.getScoreValue() > bestModel.getScoreValue()) {
                    bestModel = newModel;
                }

                // The cardinality is reversed for the next iteration to have the initial BN.
                // Given that we have previously increased the LV's cardinality, it is a new LV, so we have to access it by its name.
                clonedNet = clonedNet.increaseCardinality(clonedNet.getLatentVariable(latentVar.getName()), 1);
            }
        }

        return bestModel;
    }
}
