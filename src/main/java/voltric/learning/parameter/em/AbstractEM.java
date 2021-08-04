package voltric.learning.parameter.em;

import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.parameter.DiscreteParameterLearning;
import voltric.learning.parameter.em.config.EmConfig;
import voltric.learning.parameter.em.initialization.EmInitialization;
import voltric.learning.score.ScoreType;
import voltric.model.DiscreteBayesNet;

import java.util.HashSet;

/**
 * Created by fernando on 4/04/17.
 */
public abstract class AbstractEM implements DiscreteParameterLearning {

    /** The number of elapsed steps */
    protected int nSteps;

    /** The collection of nodes that shouldnt be updated by the EM algorithm */
    protected HashSet<String> dontUpdateNodes;

    protected ScoreType scoreType;

    protected double threshold;

    protected int nMaxSteps;

    protected boolean reuse;

    protected EmInitialization initializationMethod;

    protected int nInitCandidates;

    protected int nInitIterations;

    /**
     *
     *
     * @param config
     */
    public AbstractEM(EmConfig config){
        this.nSteps = 0;
        this.scoreType = config.getScoreType();
        this.dontUpdateNodes = config.getDontUpdateNodes();
        this.threshold = config.getThreshold();
        this.nMaxSteps = config.getnMaxSteps();
        this.reuse = config.isReuse();
        this.initializationMethod = config.getInitializationMethod();
        this.nInitCandidates = initializationMethod.getCandidates();
        this.nInitIterations = initializationMethod.getIterations();
    }

    /** {@inheritDoc} */
    public abstract LearningResult<DiscreteBayesNet> learnModel(DiscreteBayesNet bayesNet, DiscreteData dataSet);

    @Override
    public ScoreType getScoreType() {
        return scoreType;
    }

    public double getThreshold() {
        return threshold;
    }
}
