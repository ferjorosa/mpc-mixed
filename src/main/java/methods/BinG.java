package methods;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.extension.learn.parameter.InitializationTypeVBEM;
import eu.amidst.extension.learn.parameter.InitializationVBEM;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.parameter.penalizer.BishopPenalizer;
import eu.amidst.extension.learn.structure.BLFM_BinG;
import eu.amidst.extension.learn.structure.Result;
import eu.amidst.extension.learn.structure.typelocalvbem.SimpleLocalVBEM;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.distance.ChebyshevDistance;
import eu.amidst.extension.util.tuple.Tuple2;
import eu.amidst.extension.util.tuple.Tuple3;
import eu.amidst.extension.util.tuple.Tuple4;
import experiments.util.AmidstToVoltricData;
import experiments.util.AmidstToVoltricModel;
import experiments.util.EstimatePredictiveScore;
import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.learning.parameter.em.EM;
import voltric.learning.parameter.em.config.EmConfig;
import voltric.model.DiscreteBayesNet;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.*;

// NOTA: no le ponemos priors porque no las tiene en el articulo original
public class BinG implements LatentDiscreteMethod {

    private long seed;

    public BinG(long seed) {
        this.seed = seed;
    }

    public static Tuple3<BayesianNetwork, Double, Long> learnModel(DataOnMemory<DataInstance> data,
                                                                   long seed,
                                                                   LogUtils.LogLevel logLevel,
                                                                   boolean printNetwork) {

        System.out.println("\n==========================");
        System.out.println("Bin-G");
        System.out.println("==========================");

        InitializationVBEM initialVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 1, 1, false);
        VBEMConfig initialVBEMConfig = new VBEMConfig(seed, 0.01, 100, initialVBEMinitialization, new BishopPenalizer());
        InitializationVBEM localVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 4, 2, true);
        VBEMConfig localVBEMConfig = new VBEMConfig(seed, 0.01, 100, localVBEMinitialization, new BishopPenalizer());
        InitializationVBEM finalVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 64, 16, true);
        VBEMConfig finalVBEMConfig = new VBEMConfig(seed, 0.01, 100, finalVBEMinitialization, new BishopPenalizer());

        long initTime = System.currentTimeMillis();

        BLFM_BinG blfm_binG = new BLFM_BinG(3,
                new ChebyshevDistance(),
                false,
                seed,
                false,
                initialVBEMConfig,
                localVBEMConfig,
                finalVBEMConfig,
                new SimpleLocalVBEM());
        Result result = blfm_binG.learnModel(data, new HashMap<>(), logLevel);
        result.getPlateuStructure().updateParameterVariablesPrior(result.getPlateuStructure().getParameterVariablesPosterior());
        BayesianNetwork posteriorPredictive = new BayesianNetwork(result.getDag(), result.getPlateuStructure().getEFLearningBN().toConditionalDistribution());

        long endTime = System.currentTimeMillis();
        long learningTimeMs = (endTime - initTime);
        double learningTimeS = learningTimeMs / 1000;

        DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.getDefault());
        otherSymbols.setDecimalSeparator('.');
        DecimalFormat f = new DecimalFormat("0.00", otherSymbols);
        System.out.println("\n---------------------------------------------");
        System.out.println("\nELBO Score: " + f.format(result.getElbo()));
        System.out.println("Learning time (s): " + learningTimeS);
        System.out.println("Per-sample average ELBO: " + f.format(result.getElbo() / data.getNumberOfDataInstances()));
        System.out.println("Per-sample average learning time (ms): " + f.format(learningTimeMs / data.getNumberOfDataInstances()));
        System.out.println("Seed: " + seed);
        if(printNetwork)
            System.out.println("\n\n"+posteriorPredictive);

        return new Tuple3<>(posteriorPredictive, result.getElbo(), learningTimeMs);
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
        //storeLatentDiscreteModels(models, "latent_results/run_"+ run +"/discrete/"+ dataName+"/" + folds.size() + "_folds/binG", dataName, "binG");

        /* Show average time and score */
        showAverageScoreAndTime(scoresAndTimes);

        /* Store experiment results in a JSON file */
        storeResults(scoresAndTimes, "latent_results/run_"+ run +"/discrete/"+ dataName+"/" + folds.size() + "_folds/binG", dataName + "_results_binG.json");
    }

    /**
     * Runs the algorithm using k-fold cross validation.
     *
     * @return  Model, Test-LL, Train-BIC, learning time (ms)
     */
    private List<Tuple4<DiscreteBayesNet, Double, Double, Long>> runLatent(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                                                           LogUtils.LogLevel foldLogLevel) {

        System.out.println("\n==========================");
        System.out.println("Bin-G");
        System.out.println("==========================");

        InitializationVBEM initialVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 1, 1, false);
        VBEMConfig initialVBEMConfig = new VBEMConfig(seed, 0.01, 100, initialVBEMinitialization, new BishopPenalizer());
        InitializationVBEM localVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 4, 2, true);
        VBEMConfig localVBEMConfig = new VBEMConfig(seed, 0.01, 100, localVBEMinitialization, new BishopPenalizer());
        InitializationVBEM finalVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 64, 16, true);
        VBEMConfig finalVBEMConfig = new VBEMConfig(seed, 0.01, 100, finalVBEMinitialization, new BishopPenalizer());

        List<Tuple4<DiscreteBayesNet, Double, Double, Long>> foldsResults = new ArrayList<>();

        for(int i = 0; i < folds.size(); i++) {

            DataOnMemory<DataInstance> trainData = folds.get(i).getFirst();
            DataOnMemory<DataInstance> testData = folds.get(i).getSecond();

            DiscreteData trainDataVoltric = AmidstToVoltricData.transform(trainData);
            DiscreteData testDataVoltric = AmidstToVoltricData.transform(testData);

            long initTime = System.currentTimeMillis();

            BLFM_BinG blfm_binG = new BLFM_BinG(3,
                    new ChebyshevDistance(),
                    false,
                    seed,
                    false,
                    initialVBEMConfig,
                    localVBEMConfig,
                    finalVBEMConfig,
                    new SimpleLocalVBEM());
            Result result = blfm_binG.learnModel(trainData, new LinkedHashMap<>(), LogUtils.LogLevel.NONE);
            result.getPlateuStructure().updateParameterVariablesPrior(result.getPlateuStructure().getParameterVariablesPosterior());
            BayesianNetwork posteriorPredictive = new BayesianNetwork(result.getDag(), result.getPlateuStructure().getEFLearningBN().toConditionalDistribution());

            DiscreteBayesNet voltricBLTM = AmidstToVoltricModel.transform(posteriorPredictive);
            AbstractEM em = new EM(new EmConfig(seed));
            LearningResult<DiscreteBayesNet> voltricResult = em.learnModel(voltricBLTM, trainDataVoltric);
            DiscreteBayesNet voltricResultNet = voltricResult.getBayesianNetwork();

            long endTime = System.currentTimeMillis();

            double testLL = EstimatePredictiveScore.voltricLL(voltricResultNet, testDataVoltric);
            double trainBIC = EstimatePredictiveScore.voltricBIC(voltricResultNet, trainDataVoltric);
            long foldTime = (endTime - initTime);
            foldsResults.add(new Tuple4<>(voltricResultNet, testLL, trainBIC, foldTime));

            LogUtils.info("----------------------------------------", foldLogLevel);
            LogUtils.info("Fold " + (i+1) , foldLogLevel);
            LogUtils.info("Test Log-Likelihood: " + testLL, foldLogLevel);
            LogUtils.info("Train BIC: " + trainBIC, foldLogLevel);
            LogUtils.info("Time: " + foldTime + " ms", foldLogLevel);
        }

        return foldsResults;
    }
}
