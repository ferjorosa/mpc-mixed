package voltric.learning.parameter.em;

import voltric.data.DiscreteData;
import voltric.inference.CliqueTreePropagation;
import voltric.learning.parameter.em.config.EmConfig;
import voltric.learning.parameter.em.initialization.ChickeringHeckerman;
import voltric.learning.parameter.em.initialization.MultipleRestarts;
import voltric.learning.parameter.em.initialization.PyramidInitialization;
import voltric.model.DiscreteBayesNet;

/**
 * Created by fernando on 4/04/17.
 */
public abstract class AbstractSequentialEM extends AbstractEM{

    /**
     *
     * @param config
     */
    public AbstractSequentialEM(EmConfig config) {
        super(config);
    }

    /**
     * Devuelve la log-likelihood de aprender la BN con el dataSet.
     *
     * @param ctp
     * @param dataSet
     */
    protected abstract double emStep(CliqueTreePropagation ctp, DiscreteData dataSet);

    /**
     * Selects a good starting point using the Chickering & Heckerman's strategy.
     *
     * <p><b>Note:</b> that this restarting phase will terminate midway if the maximum number of steps is reached. However,
     * it will not terminate if the EM algorithm already converges on some starting point. That makes things complicated.</p>
     */
    protected abstract CliqueTreePropagation chickeringHeckermanInitialization(DiscreteBayesNet bayesNet, DiscreteData dataSet);

    protected abstract CliqueTreePropagation randomInitialization(DiscreteBayesNet bayesNet, DiscreteData dataSet);

    protected abstract CliqueTreePropagation pyramidInitialization(DiscreteBayesNet bayesNet, DiscreteData dataSet);

    /**
     * The EM initializationMethod strategy
     *
     * @param bayesNet
     * @param dataSet
     * @return
     */
    protected CliqueTreePropagation emStart(DiscreteBayesNet bayesNet, DiscreteData dataSet){
        if(this.initializationMethod instanceof PyramidInitialization)
            return pyramidInitialization(bayesNet, dataSet);
        if(this.initializationMethod instanceof ChickeringHeckerman)
            return chickeringHeckermanInitialization(bayesNet, dataSet);
        if(this.initializationMethod instanceof MultipleRestarts)
            return randomInitialization(bayesNet, dataSet);
        else
            throw new IllegalArgumentException("Invalid escape method");
    }
}
