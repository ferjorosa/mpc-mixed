package voltric.clustering.unidimensional;

import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.learning.structure.hillclimbing.global.GlobalHillClimbing;
import voltric.learning.structure.hillclimbing.global.HcOperator;
import voltric.learning.structure.hillclimbing.global.IncreaseLatentCardinality;
import voltric.model.DiscreteBayesNet;
import voltric.model.HLCM;
import voltric.model.creator.HlcmCreator;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

public class LearnLatentLCM {

    public static LearningResult<DiscreteBayesNet> learnModel(int cardinality,
                                                              DiscreteData dataSet,
                                                              AbstractEM em,
                                                              long seed){

        // Create an LCM
        HLCM initialModel = HlcmCreator.createLCM(dataSet.getVariables(), cardinality, new Random(seed));
        return em.learnModel(initialModel, dataSet);
    }

    public static LearningResult<DiscreteBayesNet> learnModel(HLCM initialModel,
                                                              DiscreteData dataSet,
                                                              AbstractEM em){

        return em.learnModel(initialModel, dataSet);
    }

    public static LearningResult<DiscreteBayesNet> learnModelToMaxCardinality(HLCM initialModel,
                                                                              DiscreteData dataSet,
                                                                              AbstractEM em,
                                                                              double threshold,
                                                                              int maxCardinality) {

        // A hill-climbing search process is applied where only the IncreaseOlcmCard operator is used
        IncreaseLatentCardinality ilcOperator = new IncreaseLatentCardinality(maxCardinality);
        Set<HcOperator> operatorSet = new HashSet<>();
        operatorSet.add(ilcOperator);

        // The maximum number of iterations is the maximum cardinality - (initCardinality - 1)
        GlobalHillClimbing hillClimbing = new GlobalHillClimbing(operatorSet, maxCardinality - 1, threshold);

        return hillClimbing.learnModel(initialModel, dataSet, em);
    }
}
