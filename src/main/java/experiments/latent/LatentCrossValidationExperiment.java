package experiments.latent;

import eu.amidst.extension.util.LogUtils;

public interface LatentCrossValidationExperiment {

    void runCrossValExperiment(long seed, int kFolds, int run, LogUtils.LogLevel foldLogLevel) throws Exception;
}
