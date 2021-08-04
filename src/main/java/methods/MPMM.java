package methods;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.extension.learn.parameter.InitializationTypeVBEM;
import eu.amidst.extension.learn.parameter.InitializationVBEM;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.parameter.penalizer.BishopPenalizer;
import eu.amidst.extension.learn.structure.BLFM_GS;
import eu.amidst.extension.learn.structure.Result;
import eu.amidst.extension.learn.structure.typelocalvbem.SimpleLocalVBEM;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.tuple.Tuple2;
import eu.amidst.extension.util.tuple.Tuple3;
import eu.amidst.extension.util.tuple.Tuple4;
import experiments.util.EstimatePredictiveScore;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;

public class MPMM implements LatentContinuousMethod {

    private long seed;

    public MPMM(long seed) {
        this.seed = seed;
    }

    public static Tuple3<BayesianNetwork, Double, Long> learnModel(DataOnMemory<DataInstance> data,
                                                                   long seed,
                                                                   LogUtils.LogLevel logLevel,
                                                                   boolean printNetwork) {

        System.out.println("\n==========================");
        System.out.println("MPMM");
        System.out.println("==========================");

        InitializationVBEM vbemInitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 64, 16, true);
        VBEMConfig vbemConfig = new VBEMConfig(seed, 0.01, 100, vbemInitialization, new BishopPenalizer());

        long initTime = System.currentTimeMillis();

        BLFM_GS gs = new BLFM_GS(vbemConfig, new SimpleLocalVBEM());
        Result result = gs.learnModel(data, new HashMap<>(), logLevel);
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
    public void runLatentContinuous(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                    String dataName,
                                    int run,
                                    LogUtils.LogLevel foldLogLevel) throws Exception {
        List<Tuple3<Double, Double, Long>> scoresAndTimes = new ArrayList<>(folds.size());

        /* Run */
        List<Tuple4<BayesianNetwork, Double, Double, Long>> results = runLatent(folds, foldLogLevel);
        for(Tuple4<BayesianNetwork, Double, Double, Long> result: results) {
            scoresAndTimes.add(new Tuple3<>(result.getSecond(), result.getThird(), result.getFourth()));
        }

        /* Show average time and score */
        showAverageScoreAndTime(scoresAndTimes);

        /* Store experiment results in a JSON file */
        storeResults(scoresAndTimes, "latent_results/run_"+ run +"/continuous/"+ dataName+"/" + folds.size()
                + "_folds/GS", dataName + "_results_MPMM.json");
    }

    private List<Tuple4<BayesianNetwork, Double, Double, Long>> runLatent(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                                                          LogUtils.LogLevel foldLogLevel) {

        System.out.println("\n==========================");
        System.out.println("MPMM");
        System.out.println("==========================");

        List<Tuple4<BayesianNetwork, Double, Double, Long>> foldsResults = new ArrayList<>();

        InitializationVBEM vbemInitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 64, 16, true);
        VBEMConfig vbemConfig = new VBEMConfig(seed, 0.01, 100, vbemInitialization, new BishopPenalizer());

        /* Iterate through the folds and learn an LCM on each one */
        for(int i = 0; i < folds.size(); i++) {
            Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>> fold = folds.get(i);

            DataOnMemory<DataInstance> trainData = fold.getFirst();
            DataOnMemory<DataInstance> testData = fold.getSecond();

            long initTime = System.currentTimeMillis();

            BLFM_GS gs = new BLFM_GS(vbemConfig, new SimpleLocalVBEM());
            Result result = gs.learnModel(trainData, new HashMap<>(), LogUtils.LogLevel.NONE);
            result.getPlateuStructure().updateParameterVariablesPrior(result.getPlateuStructure().getParameterVariablesPosterior());
            BayesianNetwork posteriorPredictive = new BayesianNetwork(result.getDag(), result.getPlateuStructure().getEFLearningBN().toConditionalDistribution());

            long endTime = System.currentTimeMillis();

            double testLL = EstimatePredictiveScore.amidstLL(posteriorPredictive, testData);
            double trainBIC = EstimatePredictiveScore.amidstBIC(posteriorPredictive, trainData);
            long foldTime = (endTime - initTime);
            foldsResults.add(new Tuple4<>(posteriorPredictive, testLL, trainBIC, foldTime));

            LogUtils.info("----------------------------------------", foldLogLevel);
            LogUtils.info("Fold " + (i+1) , foldLogLevel);
            LogUtils.info("Test Log-Likelihood: " + testLL, foldLogLevel);
            LogUtils.info("Train BIC: " + trainBIC, foldLogLevel);
            LogUtils.info("Time: " + foldTime + " ms", foldLogLevel);
        }

        return foldsResults;
    }
}
