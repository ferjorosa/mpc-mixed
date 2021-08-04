package methods;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.extension.learn.parameter.InitializationTypeVBEM;
import eu.amidst.extension.learn.parameter.InitializationVBEM;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.parameter.penalizer.BishopPenalizer;
import eu.amidst.extension.learn.structure.BLFM_IncLearnerMax;
import eu.amidst.extension.learn.structure.Result;
import eu.amidst.extension.learn.structure.operator.incremental.BlfmIncAddArc;
import eu.amidst.extension.learn.structure.operator.incremental.BlfmIncAddDiscreteNode;
import eu.amidst.extension.learn.structure.operator.incremental.BlfmIncOperator;
import eu.amidst.extension.learn.structure.typelocalvbem.TypeLocalVBEM;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.tuple.Tuple2;
import eu.amidst.extension.util.tuple.Tuple3;
import eu.amidst.extension.util.tuple.Tuple4;
import experiments.util.AmidstToVoltricModel;
import experiments.util.EstimatePredictiveScore;
import voltric.model.DiscreteBayesNet;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.*;
import java.util.stream.Collectors;

public class IL implements LatentDiscreteMethod, LatentContinuousMethod, LatentMixedMethod, BayesianMethod {

    private long seed;
    private boolean iterationGlobalVBEM;
    private boolean allowObservedToObserved;
    private boolean allowObservedToLatent;
    private Map<String, double[]> priors;
    private TypeLocalVBEM typeLocalVBEM;

    public IL(long seed,
              boolean iterationGlobalVBEM,
              boolean allowObservedToObserved,
              boolean allowObservedToLatent,
              TypeLocalVBEM typeLocalVBEM) {
        this.seed = seed;
        this.iterationGlobalVBEM = iterationGlobalVBEM;
        this.allowObservedToLatent = allowObservedToLatent;
        this.allowObservedToObserved = allowObservedToObserved;
        this.priors = new HashMap<>();
        this.typeLocalVBEM = typeLocalVBEM;
    }

    public static Tuple3<BayesianNetwork, Double, Long> learnModel(DataOnMemory<DataInstance> data,
                                                                   Map<String, double[]> priors,
                                                                   long seed,
                                                                   boolean iterationGlobalVBEM,
                                                                   boolean allowObservedToObserved,
                                                                   boolean allowObservedToLatent,
                                                                   TypeLocalVBEM typeLocalVBEM,
                                                                   LogUtils.LogLevel logLevel,
                                                                   boolean printNetwork) {

        System.out.println("\n==========================");
        System.out.println("IL (allowObservedToObserved = " + allowObservedToObserved + ", allowObservedToLatent = "+allowObservedToLatent+") ");
        System.out.println("==========================");

        InitializationVBEM initialVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 1, 1, false);
        VBEMConfig initialVBEMConfig = new VBEMConfig(seed, 0.01, 100, initialVBEMinitialization, new BishopPenalizer());
        InitializationVBEM localVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 4, 2, true);
        VBEMConfig localVBEMConfig = new VBEMConfig(seed, 0.01, 100, localVBEMinitialization, new BishopPenalizer());
        InitializationVBEM iterationVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 16, 4, true);
        VBEMConfig iterationVBEMConfig = new VBEMConfig(seed, 0.01, 100, iterationVBEMinitialization, new BishopPenalizer());
        InitializationVBEM finalVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 64, 16, true);
        VBEMConfig finalVBEMConfig = new VBEMConfig(seed, 0.01, 100, finalVBEMinitialization, new BishopPenalizer());

        long initTime = System.currentTimeMillis();

        Set<BlfmIncOperator> operators = new LinkedHashSet<>();
        BlfmIncAddDiscreteNode addDiscreteNodeOperator = new BlfmIncAddDiscreteNode(2, Integer.MAX_VALUE, localVBEMConfig, typeLocalVBEM);
        BlfmIncAddArc addArcOperator = new BlfmIncAddArc(allowObservedToObserved, allowObservedToLatent, allowObservedToObserved, localVBEMConfig, typeLocalVBEM);
        operators.add(addDiscreteNodeOperator);
        operators.add(addArcOperator);

        BLFM_IncLearnerMax incLearnerMax = new BLFM_IncLearnerMax(operators,
                iterationGlobalVBEM,
                initialVBEMConfig,
                localVBEMConfig,
                iterationVBEMConfig,
                finalVBEMConfig,
                typeLocalVBEM);

        Result result = incLearnerMax.learnModel(data, priors, logLevel);
        result.getPlateuStructure().updateParameterVariablesPrior(result.getPlateuStructure().getParameterVariablesPosterior());
        BayesianNetwork posteriorPredictive = new BayesianNetwork(result.getDag(), result.getPlateuStructure().getEFLearningBN().toConditionalDistribution());

        long endTime = System.currentTimeMillis();
        long learningTimeMs = (endTime - initTime);
        double learningTimeS = learningTimeMs / 1000;

        DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.getDefault());
        otherSymbols.setDecimalSeparator('.');
        DecimalFormat f = new DecimalFormat("0.00", otherSymbols);
        System.out.println("\nELBO Score: " + f.format(result.getElbo()));
        System.out.println("Learning time (s): " + learningTimeS);
        System.out.println("Per-sample average ELBO: " + f.format(result.getElbo() / data.getNumberOfDataInstances()));
        System.out.println("Per-sample average learning time (ms): " + f.format(learningTimeMs / data.getNumberOfDataInstances()));
        System.out.println("Seed: " + seed);
        if(printNetwork)
            System.out.println("\n\n"+posteriorPredictive);

        return new Tuple3<>(posteriorPredictive, result.getElbo(), learningTimeMs);

    }


    /* The priors can be different for each dataSet (i.e., Empirical Bayes) */
    @Override
    public void setPriors(Map<String, double[]> priors) {
        this.priors = priors;
    }

    @Override
    public void runLatentDiscrete(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                  String dataName,
                                  int run,
                                  LogUtils.LogLevel foldLogLevel) throws Exception {

        List<DiscreteBayesNet> models = new ArrayList<>(folds.size());
        List<Tuple3<Double, Double, Long>> scores = new ArrayList<>(folds.size());

        /* Run */
        List<Tuple4<BayesianNetwork, Double, Double, Long>> results = runLatent(folds, foldLogLevel);
        for(Tuple4<BayesianNetwork, Double, Double, Long> result: results) {
            models.add(AmidstToVoltricModel.transform(result.getFirst()));
            scores.add(new Tuple3<>(result.getSecond(), result.getThird(), result.getFourth()));
        }

        /* Store models */
//        storeLatentDiscreteModels(models, "latent_results/run_"+ run +"/discrete/"+ dataName + "/" + folds.size()
//                + "_folds/IL", dataName, "IL");

        /* Show average time and score */
        showAverageScoreAndTime(scores);

        /* Store experiment results in a JSON file */
        storeResults(scores, "latent_results/run_"+ run +"/discrete/"+ dataName+"/" + folds.size()
                        + "_folds/IL", dataName + "_results_IL.json");
    }

    @Override
    public void runLatentContinuous(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                    String dataName,
                                    int run,
                                    LogUtils.LogLevel foldLogLevel) throws Exception {

        List<Tuple3<Double, Double, Long>> scores = new ArrayList<>(folds.size());

        /* Run */
        List<Tuple4<BayesianNetwork, Double, Double, Long>> results = runLatent(folds, foldLogLevel);
        for(Tuple4<BayesianNetwork, Double, Double, Long> result: results) {
            scores.add(new Tuple3<>(result.getSecond(), result.getThird(), result.getFourth()));
        }

        /* Show average time and score */
        showAverageScoreAndTime(scores);

        /* Store experiment results in a JSON file */
        storeResults(scores, "latent_results/run_"+ run +"/continuous/"+ dataName+"/" + folds.size()
                        + "_folds/IL", dataName + "_results_IL.json");
    }

    @Override
    public void runLatentMixed(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                               String dataName,
                               int run,
                               LogUtils.LogLevel foldLogLevel) throws Exception {

        List<Tuple3<Double, Double, Long>> scoresAndTimes = new ArrayList<>(folds.size());

        /* Run */
        List<Tuple4<BayesianNetwork, Double, Double, Long>> results = runLatent(folds, foldLogLevel);
        for(Tuple4<BayesianNetwork, Double, Double, Long> variationalResult: results) {
            scoresAndTimes.add(new Tuple3<>(variationalResult.getSecond(), variationalResult.getThird(), variationalResult.getFourth()));
        }
        List<BayesianNetwork> models = results.stream().map(x->x.getFirst()).collect(Collectors.toList());

        /* Store models */
//        storeMixedModels(models, "latent_results/run_"+ run +"/mixed/"+ dataName + "/" + folds.size()
//                + "_folds/IL", dataName, "IL");

        /* Show average time and score */
        showAverageScoreAndTime(scoresAndTimes);

        /* Store experiment results in a JSON file */
        storeResults(scoresAndTimes, "latent_results/run_"+ run +"/mixed/"+ dataName+"/" + folds.size()
                + "_folds/IL", dataName + "_results_IL.json");
    }

    private List<Tuple4<BayesianNetwork, Double, Double, Long>> runLatent(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                                                          LogUtils.LogLevel foldLogLevel) {
        System.out.println("\n==========================");
        System.out.println("Incremental Learner (allowObservedToObserved = " + allowObservedToObserved + ", allowObservedToLatent = "+allowObservedToLatent+") ");
        System.out.println("==========================\n");

        InitializationVBEM initialVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.RANDOM, 1, 1, false);
        VBEMConfig initialVBEMConfig = new VBEMConfig(seed, 0.01, 100, initialVBEMinitialization, new BishopPenalizer());
        InitializationVBEM localVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 4, 2, true);
        VBEMConfig localVBEMConfig = new VBEMConfig(seed, 0.01, 100, localVBEMinitialization, new BishopPenalizer());
        InitializationVBEM iterationVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 16, 4, true);
        VBEMConfig iterationVBEMConfig = new VBEMConfig(seed, 0.01, 100, iterationVBEMinitialization, new BishopPenalizer());
        InitializationVBEM finalVBEMinitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 64, 16, true);
        VBEMConfig finalVBEMConfig = new VBEMConfig(seed, 0.01, 100, finalVBEMinitialization, new BishopPenalizer());

        Set<BlfmIncOperator> operators = new LinkedHashSet<>();
        BlfmIncAddDiscreteNode addDiscreteNodeOperator = new BlfmIncAddDiscreteNode(2, Integer.MAX_VALUE, localVBEMConfig, typeLocalVBEM);
        BlfmIncAddArc addArcOperator = new BlfmIncAddArc(allowObservedToObserved, allowObservedToLatent, allowObservedToObserved, localVBEMConfig, typeLocalVBEM);
        operators.add(addDiscreteNodeOperator);
        operators.add(addArcOperator);

        BLFM_IncLearnerMax incrementalLearnerMax = new BLFM_IncLearnerMax(operators,
                iterationGlobalVBEM,
                initialVBEMConfig,
                localVBEMConfig,
                iterationVBEMConfig,
                finalVBEMConfig,
                typeLocalVBEM);

        List<Tuple4<BayesianNetwork, Double, Double, Long>> foldsResults = new ArrayList<>();

        for(int i = 0; i < folds.size(); i++) {

            DataOnMemory<DataInstance> trainData = folds.get(i).getFirst();
            DataOnMemory<DataInstance> testData = folds.get(i).getSecond();

            long initTime = System.currentTimeMillis();

            Result result = incrementalLearnerMax.learnModel(trainData, priors, LogUtils.LogLevel.NONE);
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
