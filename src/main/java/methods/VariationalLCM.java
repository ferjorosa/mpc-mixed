package methods;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variable;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.learn.parameter.InitializationTypeVBEM;
import eu.amidst.extension.learn.parameter.InitializationVBEM;
import eu.amidst.extension.learn.parameter.VBEM;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.parameter.penalizer.BishopPenalizer;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.tuple.Tuple2;
import eu.amidst.extension.util.tuple.Tuple3;
import eu.amidst.extension.util.tuple.Tuple4;
import experiments.util.AmidstToVoltricModel;
import experiments.util.EstimatePredictiveScore;
import voltric.model.DiscreteBayesNet;
import voltric.util.Tuple;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.*;
import java.util.stream.Collectors;

public class VariationalLCM implements LatentDiscreteMethod, LatentContinuousMethod, LatentMixedMethod, BayesianMethod {

    private long seed;
    private Map<String, double[]> priors;

    public VariationalLCM(long seed) {
        this.seed = seed;
        this.priors = new HashMap<>();
    }

    public static Tuple3<BayesianNetwork, Double, Long> learnModel(DataOnMemory<DataInstance> data,
                                                                   long seed,
                                                                   Map<String, double[]> priors,
                                                                   LogUtils.LogLevel logLevel,
                                                                   boolean printNetwork) {

        System.out.println("\n==========================");
        System.out.println("Variational LCM");
        System.out.println("==========================");

        InitializationVBEM vbemInitialization = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, 64, 16, false);
        VBEMConfig vbemConfig = new VBEMConfig(seed, 0.01, 100, vbemInitialization, new BishopPenalizer());

        long initTime = System.currentTimeMillis();

        Tuple<BayesianNetwork, Double> result = learnLcmToMaxCardinality(data, vbemConfig, priors, logLevel);

        long endTime = System.currentTimeMillis();
        long learningTimeMs = (endTime - initTime);
        double learningTimeS = learningTimeMs / 1000;

        DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.getDefault());
        otherSymbols.setDecimalSeparator('.');
        DecimalFormat f = new DecimalFormat("0.00", otherSymbols);
        System.out.println("\n---------------------------------------------");
        System.out.println("\nELBO Score: " + f.format(result.getSecond()));
        System.out.println("Learning time (s): " + learningTimeS);
        System.out.println("Per-sample average ELBO: " + f.format(result.getSecond() / data.getNumberOfDataInstances()));
        System.out.println("Per-sample average learning time (ms): " + f.format(learningTimeMs / data.getNumberOfDataInstances()));
        System.out.println("Seed: " + seed);
        if(printNetwork)
            System.out.println("\n\n"+result.getFirst());

        return new Tuple3<>(result.getFirst(), result.getSecond(), learningTimeMs);
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
        List<Tuple3<Double, Double, Long>> scoresAndTimes = new ArrayList<>(folds.size());

        /* Run */
        List<Tuple4<BayesianNetwork, Double, Double, Long>> results = runLatent(folds, foldLogLevel);
        for(Tuple4<BayesianNetwork, Double, Double, Long> result: results) {
            models.add(AmidstToVoltricModel.transform(result.getFirst()));
            scoresAndTimes.add(new Tuple3<>(result.getSecond(), result.getThird(), result.getFourth()));
        }

        /* Store models */
//        storeLatentDiscreteModels(models, "latent_results/run_"+ run +"/discrete/"+ dataName + "/" + folds.size()
//                + "_folds/variational_LCM" , dataName, "variational_LCM");

        /* Show average time and score */
        showAverageScoreAndTime(scoresAndTimes);

        /* Store experiment results in a JSON file */
        storeResults(scoresAndTimes, "latent_results/run_"+ run +"/discrete/"+ dataName+"/" + folds.size()
                + "_folds/variational_LCM", dataName + "_results_variational_LCM.json");
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
                + "_folds/variational_LCM",dataName + "_results_variational_LCM.json");
    }

    @Override
    public void runLatentMixed(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                               String dataName,
                               int run,
                               LogUtils.LogLevel foldLogLevel) throws Exception {

        List<Tuple3<Double, Double, Long>> scoresAndTimes = new ArrayList<>(folds.size());

        /* Run */
        List<Tuple4<BayesianNetwork, Double, Double, Long>> results = runLatent(folds, foldLogLevel);
        for(Tuple4<BayesianNetwork, Double, Double, Long> result: results) {
            scoresAndTimes.add(new Tuple3<>(result.getSecond(), result.getThird(), result.getFourth()));
        }
        List<BayesianNetwork> models = results.stream().map(x->x.getFirst()).collect(Collectors.toList());

        /* Store models */
//        storeMixedModels(models, "latent_results/run_"+ run +"/mixed/"+ dataName + "/" + folds.size()
//                + "_folds/variational_LCM" , dataName, "variational_LCM");

        /* Show average time and score */
        showAverageScoreAndTime(scoresAndTimes);

        /* Store experiment results in a JSON file */
        storeResults(scoresAndTimes, "latent_results/run_"+ run +"/mixed/"+ dataName+"/" + folds.size()
                + "_folds/variational_LCM", dataName + "_results_variational_LCM.json");

    }

    private List<Tuple4<BayesianNetwork, Double, Double, Long>> runLatent(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                                                          LogUtils.LogLevel foldLogLevel) {

        System.out.println("\n==========================");
        System.out.println("Variational LCM");
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

            Tuple<BayesianNetwork, Double> result = VariationalLCM.learnLcmToMaxCardinality(trainData, vbemConfig, priors, LogUtils.LogLevel.NONE);
            BayesianNetwork posteriorPredictive = result.getFirst();

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

    private static Tuple<BayesianNetwork, Double> learnLcmToMaxCardinality(DataOnMemory<DataInstance> data,
                                                                           VBEMConfig config,
                                                                           Map<String, double[]> priors,
                                                                           LogUtils.LogLevel logLevel) {


        VBEM vbem = new VBEM(config);
        double bestScore = -Double.MAX_VALUE;
        BayesianNetwork bestModel = null;

        for(int card = 2; card < Integer.MAX_VALUE; card++) {
            long initTime = System.currentTimeMillis();
            DAG lcmStructure = generateLcmDAG(data, "ClustVar", card);
            double currentScore = vbem.learnModelWithPriorUpdate(data, lcmStructure, priors);
            BayesianNetwork currentModel = vbem.getLearntBayesianNetwork();
            long endTime = System.currentTimeMillis();

            long learnTime = (endTime - initTime);

            LogUtils.info("\nCardinality " + card, logLevel);
            LogUtils.info("ELBO: " + currentScore, logLevel);
            LogUtils.info("Time: " + learnTime + " ms", logLevel);

            if(currentScore > bestScore) {
                bestModel = currentModel;
                bestScore = currentScore;
            } else {
                //System.out.println("SCORE STOPPED IMPROVING");
                return new Tuple<>(bestModel, bestScore);
            }
        }
        return new Tuple<>(bestModel, bestScore);
    }

    private static DAG generateLcmDAG(DataOnMemory<DataInstance> data,
                                      String latentVarName,
                                      int cardinality) {

        /* Creamos un Naive Bayes con padre latente */
        Variables variables = new Variables(data.getAttributes());
        Variable latentVar = variables.newMultinomialVariable(latentVarName, cardinality);

        DAG dag = new DAG(variables);

        for(Variable var: variables)
            if(!var.equals(latentVar))
                dag.getParentSet(var).addParent(latentVar);

        return dag;
    }
}
