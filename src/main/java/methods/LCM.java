package methods;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.tuple.Tuple2;
import eu.amidst.extension.util.tuple.Tuple3;
import eu.amidst.extension.util.tuple.Tuple4;
import experiments.util.AmidstToVoltricData;
import experiments.util.EstimatePredictiveScore;
import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.learning.parameter.em.EM;
import voltric.learning.parameter.em.config.EmConfig;
import voltric.learning.parameter.em.initialization.PyramidInitialization;
import voltric.learning.score.LearningScore;
import voltric.learning.score.ScoreType;
import voltric.model.DiscreteBayesNet;
import voltric.model.HLCM;
import voltric.model.creator.HlcmCreator;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

public class LCM implements LatentDiscreteMethod {

    private long seed;

    public LCM(long seed) {
        this.seed = seed;
    }

    @Override
    public void runLatentDiscrete(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                  String dataName,
                                  int run,
                                  LogUtils.LogLevel foldLogLevel) throws Exception {

        List<DiscreteBayesNet> models = new ArrayList<>(folds.size());
        List<Tuple3<Double, Double, Long>> scoresAndTimes = new ArrayList<>(folds.size());

        /* Run */
        List<Tuple4<DiscreteBayesNet, Double, Double, Long>> results = runLatent(folds, foldLogLevel);
        for(Tuple4<DiscreteBayesNet, Double, Double, Long> result: results) {
            models.add(result.getFirst());
            scoresAndTimes.add(new Tuple3<>(result.getSecond(), result.getThird(), result.getFourth()));
        }

        /* Store models */
//        storeLatentDiscreteModels(models, "latent_results/run_"+ run +"/discrete/"+ dataName + "/" + folds.size()
//                + "_folds/LCM" , dataName, "LCM");

        /* Show average time and score */
        showAverageScoreAndTime(scoresAndTimes);

        /* Store experiment results in a JSON file */
        storeResults(scoresAndTimes, "latent_results/run_"+ run +"/discrete/"+ dataName+"/" + folds.size()
                + "_folds/LCM", dataName + "_results_LCM.json");
    }

    /**
     * Runs the algorithm using k-fold cross validation.
     *
     * @return  Model, Test-LL, Train-BIC, learning time (ms)
     */
    private List<Tuple4<DiscreteBayesNet, Double, Double, Long>> runLatent(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                                                           LogUtils.LogLevel foldLogLevel) {

        System.out.println("\n==========================");
        System.out.println("LCM");
        System.out.println("==========================\n");

        double threshold = 1e-2;
        int nMaxSteps = 500;

        EmConfig emConfig = new EmConfig(seed, threshold, nMaxSteps, new PyramidInitialization(), false, ScoreType.BIC, new HashSet<>());
        EM em = new EM(emConfig);
        List<Tuple4<DiscreteBayesNet, Double, Double, Long>> foldsResults = new ArrayList<>();

        /* Iterate through the folds and learn an LCM on each one */
        for(int i = 0; i < folds.size(); i++) {

            Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>> fold = folds.get(i);

            DataOnMemory<DataInstance> trainData = fold.getFirst();
            DataOnMemory<DataInstance> testData = fold.getSecond();
            DiscreteData trainDataVoltric = AmidstToVoltricData.transform(trainData);
            DiscreteData testDataVoltric = AmidstToVoltricData.transform(testData);

            long initTime = System.currentTimeMillis();
            LearningResult<DiscreteBayesNet> result = learnLcmToMaxCardinality(trainDataVoltric, em, seed, threshold, LogUtils.LogLevel.NONE);
            long endTime = System.currentTimeMillis();

            DiscreteBayesNet resultNet = result.getBayesianNetwork();
            double testLL = EstimatePredictiveScore.voltricLL(resultNet, testDataVoltric);
            double trainBIC = EstimatePredictiveScore.voltricBIC(resultNet, trainDataVoltric);
            long foldTime = (endTime - initTime);
            foldsResults.add(new Tuple4<>(resultNet, testLL, trainBIC, foldTime));

            LogUtils.info("----------------------------------------", foldLogLevel);
            LogUtils.info("Fold " + (i+1) , foldLogLevel);
            LogUtils.info("Test Log-Likelihood: " + testLL, foldLogLevel);
            LogUtils.info("Train BIC: " + trainBIC, foldLogLevel);
            LogUtils.info("Time: " + foldTime + " ms", foldLogLevel);
        }

        return foldsResults;
    }

    public static Tuple4<DiscreteBayesNet, Double, Double, Long> learnModel(DataOnMemory<DataInstance> data,
                                                                            long seed,
                                                                            LogUtils.LogLevel logLevel) {

        System.out.println("\n==========================");
        System.out.println("LCM");
        System.out.println("==========================\n");

        double threshold = 1e-2;
        int nMaxSteps = 500;

        EmConfig emConfig = new EmConfig(seed, threshold, nMaxSteps, new PyramidInitialization(), true, ScoreType.BIC, new HashSet<>());
        EM em = new EM(emConfig);

        DiscreteData trainDataVoltric = AmidstToVoltricData.transform(data);

        long initTime = System.currentTimeMillis();
        LearningResult<DiscreteBayesNet> result = learnLcmToMaxCardinality(trainDataVoltric, em, seed, threshold, logLevel);
        long endTime = System.currentTimeMillis();
        long learningTimeMs = (endTime - initTime);
        double learningTimeS = learningTimeMs / 1000;

        DiscreteBayesNet resultNet = result.getBayesianNetwork();
        long learnTime = (endTime - initTime);
        double logLikelihood = LearningScore.calculateLogLikelihood(trainDataVoltric, resultNet);
        double bic = LearningScore.calculateBIC(trainDataVoltric, resultNet);
        System.out.println("\n---------------------------------------------");
        System.out.println("Log-Likelihood: " + logLikelihood);
        System.out.println("BIC: " + bic);
        System.out.println("Learning time (ms): " + learningTimeMs + " ms");
        System.out.println("Learning time (s): " + learningTimeS + " s");

        return new Tuple4<>(resultNet, logLikelihood, bic, learnTime);
    }


    private static LearningResult<DiscreteBayesNet> learnLcmToMaxCardinality(DiscreteData dataSet,
                                                                             AbstractEM em,
                                                                             long seed,
                                                                             double threshold,
                                                                             LogUtils.LogLevel logLevel) {

        LearningResult<DiscreteBayesNet> bestResult = null;

        for(int card = 2; card < Integer.MAX_VALUE; card++) {
            long initTime = System.currentTimeMillis();
            HLCM lcm = HlcmCreator.createLCM(dataSet.getVariables(), card, new Random(seed));
            LearningResult<DiscreteBayesNet> result = em.learnModel(lcm, dataSet);
            long endTime = System.currentTimeMillis();

            long learnTime = (endTime - initTime);
            double currentlogLikelihood = LearningScore.calculateLogLikelihood(dataSet, result.getBayesianNetwork());
            double currentBic = LearningScore.calculateBIC(dataSet, result.getBayesianNetwork());

            LogUtils.info("\nCardinality " + card, logLevel);
            LogUtils.info("Log-Likelihood: " + currentlogLikelihood, logLevel);
            LogUtils.info("BIC: " + currentBic, logLevel);
            LogUtils.info("Time: " + learnTime + " ms", logLevel);

            if(bestResult == null || currentBic >= bestResult.getScoreValue()) {
                bestResult = result;
            } else {
                LogUtils.info("\nSCORE STOPPED IMPROVING", logLevel);
                return bestResult;
            }
        }

        return bestResult;
    }
}
