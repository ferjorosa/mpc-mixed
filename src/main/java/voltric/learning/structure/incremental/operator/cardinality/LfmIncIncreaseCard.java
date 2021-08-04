package voltric.learning.structure.incremental.operator.cardinality;

import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.LocalEM;
import voltric.learning.parameter.em.config.EmConfig;
import voltric.learning.structure.incremental.localemtype.TypeLocalEM;
import voltric.model.DiscreteBayesNet;
import voltric.variables.DiscreteVariable;

import java.util.List;
import java.util.Random;
import java.util.Set;

/*
 * Operador de incrementar la cardinalidad donde se actualizan los parametros con el LocalEM.
 * Es un operador complementario en LFM_IncLearner, el currentSet representa aquellas variables que se va a considerar
 * un incrementao de la cardinalidad.
 * */
public class LfmIncIncreaseCard {

    private EmConfig emConfig;

    private TypeLocalEM typeLocalEM;

    public LfmIncIncreaseCard(EmConfig emConfig, TypeLocalEM typeLocalEM) {
        this.emConfig = emConfig;
        this.typeLocalEM = typeLocalEM;
    }

    public LearningResult<DiscreteBayesNet> apply(List<String> currentset,
                                                  DiscreteBayesNet bayesNet,
                                                  DiscreteData data) {

        for(String varName: currentset)
            if(!bayesNet.getNode(varName).getVariable().isLatentVariable())
                throw new IllegalArgumentException("Only latent variables are allowed in the IncreaseCard operator");

        LearningResult<DiscreteBayesNet> bestResult = new LearningResult<>(null, -Double.MAX_VALUE, emConfig.getScoreType());

        /* Iterate through all the variables in the current set */
        for(String varName: currentset){

            /* The BN is copied to avoid modifying current object */
            DiscreteBayesNet clonedNet = bayesNet.clone();

            /* Increase the selected variable's cardinality */
            clonedNet = clonedNet.increaseCardinality(clonedNet.getLatentVariable(varName), 1, new Random(emConfig.getSeed()));
            DiscreteVariable increasedCardVariable = clonedNet.getLatentVariable(varName);

            /* Select the markov blanket of the variable and locally learn the parameters */
            Set<DiscreteVariable> localVariables = typeLocalEM.variablesToUpdate(clonedNet.getNode(increasedCardVariable));
            LocalEM localEM = new LocalEM(localVariables, this.emConfig);
            LearningResult<DiscreteBayesNet> result = localEM.learnModel(clonedNet, data);

            if(result.getScoreValue() > bestResult.getScoreValue())
                bestResult = result;
        }

        return bestResult;
    }
}
