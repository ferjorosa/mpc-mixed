package experiments.latent;

import eu.amidst.extension.util.LogUtils;

public interface LatentLearnExperiment {

    void runLearnExperiment(long seed, LogUtils.LogLevel logLevel) throws Exception;
}
