package voltric.learning.parameter;

import voltric.data.DiscreteData;
import voltric.learning.parameter.em.EM;
import voltric.learning.parameter.em.ParallelEM;
import voltric.learning.parameter.em.config.EmConfig;
import voltric.learning.parameter.mle.StaticMLE;
import voltric.model.DiscreteBayesNet;

/**
 * This class as an user's interface to a set of parameter learning methods.
 *
 * TODO: Los metodos deberian devolver un valor double que seria la LogLikelihood
 */
public class ParameterLearner {

    /**
     * Executes the Expectation-Maximization (EM) algorithm to learn the parameters of the argument bayes net with respect to
     * the argument dataSet. To execute the EM algorithm is necessary to specify a set o parameters. Those parameters
     * will be specified using the {@code EmConfig} object.
     *
     * @param bayesNet the bayesian network whose parameters are going to be learned.
     * @param dataSet the dataSet used to learn the Bayesian network.
     * @param emConfig EM configuration parameters.
     */
    public static void computeEM(DiscreteBayesNet bayesNet, DiscreteData dataSet, EmConfig emConfig) {
        EM emLearner = new EM(emConfig);
        emLearner.learnModel(bayesNet, dataSet);
    }

    /**
     * Parallel execution of the Expectation-Maximization (EM) algorithm to learn the parameters of the argument bayes net
     * with respect to the argument dataSet. To execute the EM algorithm is necessary to specify a set o parameters.
     * Those parameters will be specified using the {@code EmConfig} object.
     *
     * @param bayesNet the bayesian network whose parameters are going to be learned.
     * @param dataSet the dataSet used to learn the Bayesian network.
     * @param emConfig EM configuration parameters.
     */
    public static void computeParallelEM(DiscreteBayesNet bayesNet, DiscreteData dataSet, EmConfig emConfig) {
        ParallelEM emLearner = new ParallelEM(emConfig);
        emLearner.learnModel(bayesNet, dataSet);
    }

    /**
     * Executes the StaticMLE method, which learns the parameters of the Bayesian network whose associated data observations
     * are complete. Therefore no latent nodes or incomplete dataSets are accepted.
     *
     * @param bayesNet the bayesian network whose parameters are going to be learned.
     * @param dataSet the dataSet used to learn the Bayesian network.
     */
    public static void computeMLE(DiscreteBayesNet bayesNet, DiscreteData dataSet){
        StaticMLE.computeMle(bayesNet,dataSet);
    }
}
