package methods;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.learn.structure.hillclimber.BayesianHc;
import eu.amidst.extension.learn.structure.hillclimber.BayesianHcConfig;
import eu.amidst.extension.learn.structure.hillclimber.BayesianHcResult;
import eu.amidst.extension.learn.structure.hillclimber.operator.BayesianHcAddArc;
import eu.amidst.extension.learn.structure.hillclimber.operator.BayesianHcOperator;
import eu.amidst.extension.learn.structure.hillclimber.operator.BayesianHcRemoveArc;
import eu.amidst.extension.learn.structure.hillclimber.operator.BayesianHcReverseArc;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.tuple.Tuple2;
import eu.amidst.extension.util.tuple.Tuple3;
import eu.amidst.extension.util.tuple.Tuple4;
import experiments.util.EstimatePredictiveScore;

import java.util.*;

/** Classic Hill-climbing algorithm with ELBO score */
public class HC implements LatentDiscreteMethod, LatentContinuousMethod, LatentMixedMethod, BayesianMethod {

    private BayesianHcConfig bayesianHcConfig;

    private int maxParentsHc;

    /* Initial priors */
    private Map<String, double[]> priors;

    public HC() {
        this.maxParentsHc = 3;
        this.bayesianHcConfig = new BayesianHcConfig();
    }

    public HC(BayesianHcConfig config, int maxParentsHc) {
        this.bayesianHcConfig = config;
        this.maxParentsHc = maxParentsHc;
    }

    @Override
    public void setPriors(Map<String, double[]> priors) {
        this.priors = priors;
    }

    private List<Tuple4<BayesianNetwork, Double, Double, Long>> run(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                                                    LogUtils.LogLevel foldLogLevel) {

        System.out.println("\n==========================");
        System.out.println("HC");
        System.out.println("==========================");

        List<Tuple4<BayesianNetwork, Double, Double, Long>> foldsResults = new ArrayList<>();

        for(int i = 0; i < folds.size(); i++) {

            /* Get fold data */
            DataOnMemory<DataInstance> trainData = folds.get(i).getFirst();
            DataOnMemory<DataInstance> testData = folds.get(i).getSecond();

            Variables variables = new Variables(trainData.getAttributes());
            DAG initialDag = new DAG(variables);

            BayesianHcAddArc bayesianHcAddArc = new BayesianHcAddArc(bayesianHcConfig, variables, maxParentsHc);
            BayesianHcRemoveArc bayesianHcRemoveArc = new BayesianHcRemoveArc(bayesianHcConfig, new HashMap<>());
            BayesianHcReverseArc bayesianHcReverseArc = new BayesianHcReverseArc(bayesianHcConfig, variables, maxParentsHc);
            Set<BayesianHcOperator> bayesianHcOperators = new LinkedHashSet<>(3);
            bayesianHcOperators.add(bayesianHcAddArc);
            bayesianHcOperators.add(bayesianHcRemoveArc);
            bayesianHcOperators.add(bayesianHcReverseArc);
            BayesianHc bayesianHc = new BayesianHc(this.bayesianHcConfig, 100, 0.01, bayesianHcOperators);

            long initTime = System.currentTimeMillis();
            BayesianHcResult result = bayesianHc.learnModel(initialDag, trainData, priors, foldLogLevel);
            long endTime = System.currentTimeMillis();

            BayesianNetwork posteriorPredictive = result.getBayesianNetwork();

            /* Estimate Fold scores */
            double testLL = EstimatePredictiveScore.amidstLL(posteriorPredictive, testData);
            double trainBIC = EstimatePredictiveScore.amidstBIC(posteriorPredictive, trainData);
            long foldTime = (endTime - initTime);
            foldsResults.add(new Tuple4<>(posteriorPredictive, testLL, trainBIC, foldTime));

            LogUtils.info("----------------------------------------", foldLogLevel);
            LogUtils.info("Fold " + (i+1) , foldLogLevel);
            LogUtils.info("Test Log-Likelihood: " + testLL, foldLogLevel);
            LogUtils.info("Train BIC: " + trainBIC, foldLogLevel);
            LogUtils.info("Time: " + foldTime + " ms", foldLogLevel);
            LogUtils.info("----------------------------------------", foldLogLevel);
        }

        return foldsResults;
    }

    @Override
    public void runLatentContinuous(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                    String dataName,
                                    int run,
                                    LogUtils.LogLevel foldLogLevel) throws Exception {

        List<Tuple3<Double, Double, Long>> scores = new ArrayList<>(folds.size());

        /* Run */
        List<Tuple4<BayesianNetwork, Double, Double, Long>> results = run(folds, foldLogLevel);
        for(Tuple4<BayesianNetwork, Double, Double, Long> result: results) {
            scores.add(new Tuple3<>(result.getSecond(), result.getThird(), result.getFourth()));
        }

        /* Show average time and score */
        showAverageScoreAndTime(scores);

        /* Store experiment results in a JSON file */
        storeResults(scores, "latent_results/run_"+ run +"/continuous/"+ dataName+"/" + folds.size()
                + "_folds/HC", dataName + "_results_HC.json");
    }

    @Override
    public void runLatentDiscrete(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                  String dataName,
                                  int run,
                                  LogUtils.LogLevel foldLogLevel) throws Exception {

        List<Tuple3<Double, Double, Long>> scores = new ArrayList<>(folds.size());

        /* Run */
        List<Tuple4<BayesianNetwork, Double, Double, Long>> results = run(folds, foldLogLevel);
        for(Tuple4<BayesianNetwork, Double, Double, Long> result: results) {
            scores.add(new Tuple3<>(result.getSecond(), result.getThird(), result.getFourth()));
        }

        /* Show average time and score */
        showAverageScoreAndTime(scores);

        /* Store experiment results in a JSON file */
        storeResults(scores, "latent_results/run_"+ run +"/discrete/"+ dataName+"/" + folds.size()
                + "_folds/HC", dataName + "_results_HC.json");
    }

    @Override
    public void runLatentMixed(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                               String dataName,
                               int run,
                               LogUtils.LogLevel foldLogLevel) throws Exception {

        List<Tuple3<Double, Double, Long>> scores = new ArrayList<>(folds.size());

        /* Run */
        List<Tuple4<BayesianNetwork, Double, Double, Long>> results = run(folds, foldLogLevel);
        for(Tuple4<BayesianNetwork, Double, Double, Long> result: results) {
            scores.add(new Tuple3<>(result.getSecond(), result.getThird(), result.getFourth()));
        }

        /* Show average time and score */
        showAverageScoreAndTime(scores);

        /* Store experiment results in a JSON file */
        storeResults(scores, "latent_results/run_"+ run +"/mixed/"+ dataName+"/" + folds.size()
                + "_folds/HC", dataName + "_results_HC.json");
    }
}
