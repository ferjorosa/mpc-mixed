package methods;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.learn.parameter.InitializationTypeVBEM;
import eu.amidst.extension.learn.parameter.InitializationVBEM;
import eu.amidst.extension.learn.parameter.VBEM;
import eu.amidst.extension.learn.parameter.VBEMConfig;
import eu.amidst.extension.learn.parameter.penalizer.BishopPenalizer;
import eu.amidst.extension.learn.structure.glsl.GLSL;
import eu.amidst.extension.learn.structure.glsl.operator.*;
import eu.amidst.extension.learn.structure.hillclimber.BayesianHcConfig;
import eu.amidst.extension.learn.structure.typelocalvbem.SimpleLocalVBEM;
import eu.amidst.extension.learn.structure.vbsem.InitializationTypeVBSEM;
import eu.amidst.extension.learn.structure.vbsem.InitializationVBSEM;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.tuple.Tuple2;
import eu.amidst.extension.util.tuple.Tuple3;
import eu.amidst.extension.util.tuple.Tuple4;
import experiments.util.EstimatePredictiveScore;
import experiments.util.latent.LatentFoldResult;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.util.*;
import java.util.stream.Collectors;

/*
* NOTE: cuidado con la inicializacion del GLSL en el caso missing ya que el CIL no se encuentra actualmente preparado para datos missing
*
* */
public class GLSL_Algorithm implements LatentDiscreteMethod, LatentContinuousMethod, LatentMixedMethod, BayesianMethod {

    /* Initial model for GLSL */
    public enum Initialization {
        EMPTY,
        CIL_1,
        CIL_10
    }

    private long seed;

    /* Numero maximo de padres que puede tener una variable latente */
    private int maxNumberParents_latent;

    /* Numero maximo de padres que puede tener una variable observada*/
    private int maxNumberParents_observed;

    /* Numero maximo de variables latentes en el modelo */
    private int maxNumberOfDiscreteLatentNodes;

    /* Numero de candidatos para el VBEM */
    private int nVbemCandidates;

    /* Numero de candidatos para el VBSEM */
    private int nVbsemCandidates;

    /* Initial model for GLSL */
    private Initialization initialization;

    /* Initial priors */
    private Map<String, double[]> priors;

    public GLSL_Algorithm(long seed,
                          int maxNumberParents_latent,
                          int maxNumberParents_observed,
                          int maxNumberOfDiscreteLatentNodes,
                          int nVbemCandidates,
                          int nVbsemCandidates,
                          Initialization initialization) {
        this.seed = seed;
        this.maxNumberParents_latent = maxNumberParents_latent;
        this.maxNumberParents_observed = maxNumberParents_observed;
        this.maxNumberOfDiscreteLatentNodes = maxNumberOfDiscreteLatentNodes;
        this.nVbemCandidates = nVbemCandidates;
        this.nVbsemCandidates = nVbsemCandidates;
        this.initialization = initialization;
    }

    @Override
    public void setPriors(Map<String, double[]> priors) {
        this.priors = priors;
    }

    private List<Tuple4<BayesianNetwork, Double, Double, Long>> run(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                                                    LogUtils.LogLevel foldLogLevel) {

        System.out.println("\n==========================");
        System.out.println("GLSL (" + this.initialization + ")");
        System.out.println("==========================\n");

        /**************************************************************************************************************/
        InitializationVBEM initializationVBEM = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, nVbemCandidates, nVbemCandidates/2, true);
        VBEMConfig vbemConfig = new VBEMConfig(seed, 0.01, 100, initializationVBEM, new BishopPenalizer());
        InitializationVBSEM initializationVBSEM = new InitializationVBSEM(InitializationTypeVBSEM.NONE, nVbsemCandidates, nVbsemCandidates/2, 0.2, true);
        BayesianHcConfig bayesianHcConfig = new BayesianHcConfig(seed, 0.01, 100);

        LatentVarCounter latentVarCounter = new LatentVarCounter();
        GLSL_AddLatent_child glsl_addLatent_child = new GLSL_AddLatent_child(maxNumberOfDiscreteLatentNodes, maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM, latentVarCounter);
        GLSL_AddLatent_ind glsl_addLatent_ind = new GLSL_AddLatent_ind(maxNumberOfDiscreteLatentNodes, 2, maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM, latentVarCounter);
        GLSL_RemoveLatent glsl_removeLatent = new GLSL_RemoveLatent(maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM);
        GLSL_IncreaseCard glsl_increaseCard = new GLSL_IncreaseCard(Integer.MAX_VALUE,maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM);
        GLSL_DecreaseCard glsl_decreaseCard = new GLSL_DecreaseCard(2, maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM);

        Set<GLSL_Operator> glsl_operators = new LinkedHashSet<>();
        glsl_operators.add(glsl_addLatent_child);
        glsl_operators.add(glsl_addLatent_ind);
        glsl_operators.add(glsl_removeLatent);
        glsl_operators.add(glsl_increaseCard);
        glsl_operators.add(glsl_decreaseCard);

        GLSL glsl = new GLSL(Integer.MAX_VALUE, glsl_operators);

        /**************************************************************************************************************/
        List<Tuple4<BayesianNetwork, Double, Double, Long>> foldsResults = new ArrayList<>();

        for(int i = 0; i < folds.size(); i++) {

            /* Get fold data */
            DataOnMemory<DataInstance> trainData = folds.get(i).getFirst();
            DataOnMemory<DataInstance> testData = folds.get(i).getSecond();

            /* Initial model */
            Tuple2<BayesianNetwork, Double> initialResult = initialize(trainData, vbemConfig);

            /* Learn model using GLSL */
            long initTime = System.currentTimeMillis();
            Tuple2<BayesianNetwork, Double> glsl_result = glsl.learnModel(initialResult.getFirst(), initialResult.getSecond(), trainData, LogUtils.LogLevel.INFO, LogUtils.LogLevel.INFO);
            BayesianNetwork posteriorPredictive = glsl_result.getFirst();
            long endTime = System.currentTimeMillis();

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
        }

        return foldsResults;
    }

    private Tuple2<BayesianNetwork, Double> initialize(DataOnMemory<DataInstance> data,
                                                       VBEMConfig vbemConfig) {

        switch (this.initialization) {
            case EMPTY:
                Variables variables = new Variables(data.getAttributes());
                DAG emptyDag = new DAG(variables);
                VBEM vbem = new VBEM(vbemConfig);
                double emptyDagScore = vbem.learnModelWithPriorUpdate(data, emptyDag);
                BayesianNetwork emptyBn = vbem.getLearntBayesianNetwork();
                return new Tuple2<>(emptyBn, emptyDagScore);

            case CIL_1:
                Tuple3<BayesianNetwork, Double, Long> result_cil_1 = CIL.learnModel(data,
                        priors,
                        seed,
                        1,
                        true,
                        true,
                        true,
                        3,
                        false,
                        false,
                        new SimpleLocalVBEM(),
                        LogUtils.LogLevel.NONE,
                        false);
                return new Tuple2<>(result_cil_1.getFirst(), result_cil_1.getSecond());

            case CIL_10:
                Tuple3<BayesianNetwork, Double, Long> result_cil_10 = CIL.learnModel(data,
                        priors,
                        seed,
                        10,
                        true,
                        true,
                        true,
                        3,
                        false,
                        false,
                        new SimpleLocalVBEM(),
                        LogUtils.LogLevel.NONE,
                        false);
                return new Tuple2<>(result_cil_10.getFirst(), result_cil_10.getSecond());

            default:
                throw new IllegalStateException("Invalid GLSL initialization");
        }
    }

    @Override
    public void runLatentDiscrete(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                  String dataName,
                                  int run,
                                  LogUtils.LogLevel foldLogLevel) throws Exception {

        List<Tuple3<Double, Double, Long>> scoresAndTimes = new ArrayList<>(folds.size());

        /* Run */
        List<Tuple4<BayesianNetwork, Double, Double, Long>> results = run(folds, foldLogLevel);
        for(Tuple4<BayesianNetwork, Double, Double, Long> variationalResult: results) {
            scoresAndTimes.add(new Tuple3<>(variationalResult.getSecond(), variationalResult.getThird(), variationalResult.getFourth()));
        }
        List<BayesianNetwork> models = results.stream().map(x->x.getFirst()).collect(Collectors.toList());

        /* Store models */
//        storeMixedModels(models, "latent_results/run_"+ run +"/discrete/"+ dataName + "/" + folds.size()
//                + "_folds/GLSL", dataName, "GLSL");

        /* Show average time and score */
        showAverageScoreAndTime(scoresAndTimes);

        /* Store experiment results in a JSON file */
        storeResults(scoresAndTimes, "latent_results/run_"+ run +"/discrete/"+ dataName+"/" + folds.size()
                + "_folds/GLSL_" + this.initialization, dataName + "_results_GLSL_"+this.initialization+".json");
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
                + "_folds/GLSL_" + this.initialization, dataName + "_results_GLSL_"+this.initialization+".json");
    }

    @Override
    public void runLatentMixed(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                               String dataName,
                               int run,
                               LogUtils.LogLevel foldLogLevel) throws Exception {

        List<Tuple3<Double, Double, Long>> scoresAndTimes = new ArrayList<>(folds.size());

        /* Run */
        List<Tuple4<BayesianNetwork, Double, Double, Long>> results = run(folds, foldLogLevel);
        for(Tuple4<BayesianNetwork, Double, Double, Long> variationalResult: results) {
            scoresAndTimes.add(new Tuple3<>(variationalResult.getSecond(), variationalResult.getThird(), variationalResult.getFourth()));
        }
        List<BayesianNetwork> models = results.stream().map(x->x.getFirst()).collect(Collectors.toList());

        /* Store models */
//        storeMixedModels(models, "latent_results/run_"+ run +"/mixed/"+ dataName + "/" + folds.size()
//                + "_folds/GLSL", dataName, "GLSL");

        /* Show average time and score */
        showAverageScoreAndTime(scoresAndTimes);

        /* Store experiment results in a JSON file */
        storeResults(scoresAndTimes, "latent_results/run_"+ run +"/mixed/"+ dataName+"/" + folds.size()
                + "_folds/GLSL_"+this.initialization, dataName + "_results_GLSL_"+this.initialization+".json");
    }

    /******************************************************************************************************************/
    /******************************************************************************************************************/
    /******************************************************************************************************************/
    private Tuple4<BayesianNetwork, Double, Double, Long> runFold(Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>> fold,
                                                                  int foldNumber,
                                                                  LogUtils.LogLevel foldLogLevel) {

        System.out.println("\n==========================");
        System.out.println("GLSL (" + this.initialization + ")");
        System.out.println("==========================\n");
        System.out.println("Fold " + foldNumber);

        /**************************************************************************************************************/
        InitializationVBEM initializationVBEM = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, nVbemCandidates, nVbemCandidates/2, true);
        VBEMConfig vbemConfig = new VBEMConfig(seed, 0.01, 100, initializationVBEM, new BishopPenalizer());
        InitializationVBSEM initializationVBSEM = new InitializationVBSEM(InitializationTypeVBSEM.NONE, nVbsemCandidates, nVbsemCandidates/2, 0.2, true);
        BayesianHcConfig bayesianHcConfig = new BayesianHcConfig(seed, 0.01, 100);

        LatentVarCounter latentVarCounter = new LatentVarCounter();
        GLSL_AddLatent_child glsl_addLatent_child = new GLSL_AddLatent_child(maxNumberOfDiscreteLatentNodes, maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM, latentVarCounter);
        GLSL_AddLatent_ind glsl_addLatent_ind = new GLSL_AddLatent_ind(maxNumberOfDiscreteLatentNodes, 2, maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM, latentVarCounter);
        GLSL_RemoveLatent glsl_removeLatent = new GLSL_RemoveLatent(maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM);
        GLSL_IncreaseCard glsl_increaseCard = new GLSL_IncreaseCard(Integer.MAX_VALUE,maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM);
        GLSL_DecreaseCard glsl_decreaseCard = new GLSL_DecreaseCard(2, maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM);

        Set<GLSL_Operator> glsl_operators = new LinkedHashSet<>();
        glsl_operators.add(glsl_addLatent_child);
        glsl_operators.add(glsl_addLatent_ind);
        glsl_operators.add(glsl_removeLatent);
        glsl_operators.add(glsl_increaseCard);
        glsl_operators.add(glsl_decreaseCard);

        GLSL glsl = new GLSL(Integer.MAX_VALUE, glsl_operators);

        /* Get fold data */
        DataOnMemory<DataInstance> trainData = fold.getFirst();
        DataOnMemory<DataInstance> testData = fold.getSecond();

        /* Initial model */
        Tuple2<BayesianNetwork, Double> initialResult = initialize(trainData, vbemConfig);

        /* Learn model using GLSL */
        long initTime = System.currentTimeMillis();
        Tuple2<BayesianNetwork, Double> glsl_result = glsl.learnModel(initialResult.getFirst(), initialResult.getSecond(), trainData, LogUtils.LogLevel.INFO, LogUtils.LogLevel.INFO);
        BayesianNetwork posteriorPredictive = glsl_result.getFirst();
        long endTime = System.currentTimeMillis();

        /* Estimate Fold scores */
        double testLL = EstimatePredictiveScore.amidstLL(posteriorPredictive, testData);
        double trainBIC = EstimatePredictiveScore.amidstBIC(posteriorPredictive, trainData);
        long foldTime = (endTime - initTime);
        Tuple4<BayesianNetwork, Double, Double, Long> foldResult = new Tuple4<>(posteriorPredictive, testLL, trainBIC, foldTime);

        LogUtils.info("----------------------------------------", foldLogLevel);
        LogUtils.info("Fold " + (foldNumber) , foldLogLevel);
        LogUtils.info("Test Log-Likelihood: " + testLL, foldLogLevel);
        LogUtils.info("Train BIC: " + trainBIC, foldLogLevel);
        LogUtils.info("Time: " + foldTime + " ms", foldLogLevel);

        return foldResult;
    }

    public void runDiscreteFold(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                 int foldNumber,
                                 String dataName,
                                 int run,
                                 LogUtils.LogLevel foldLogLevel) throws Exception {

        /* Run fold */
        Tuple4<BayesianNetwork, Double, Double, Long> foldResult = runFold(folds.get(foldNumber - 1), foldNumber, foldLogLevel);

        /* Store fold result in a JSON file */
        storeFoldResult(foldResult,
                "latent_results/run_" + run + "/discrete/" + dataName + "/" + folds.size() + "_folds/GLSL_"+this.initialization,
                dataName + "_results_GLSL_"+this.initialization + "_"+  foldNumber  +".json");
    }

    public void runContinuousFold(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                   int foldNumber,
                                   String dataName,
                                   int run,
                                   LogUtils.LogLevel foldLogLevel) throws Exception {
        /* Run fold */
        Tuple4<BayesianNetwork, Double, Double, Long> foldResult = runFold(folds.get(foldNumber - 1), foldNumber, foldLogLevel);

        /* Store fold result in a JSON file */
        storeFoldResult(foldResult,
                "latent_results/run_" + run + "/continuous/" + dataName + "/" + folds.size() + "_folds/GLSL_"+this.initialization,
                dataName + "_results_GLSL_"+this.initialization + "_"+  foldNumber  +".json");
    }

    public void runMixedFold(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                              int foldNumber,
                              String dataName,
                              int run,
                              LogUtils.LogLevel foldLogLevel) throws Exception {
        /* Run fold */
        Tuple4<BayesianNetwork, Double, Double, Long> foldResult = runFold(folds.get(foldNumber - 1), foldNumber, foldLogLevel);

        /* Store fold result in a JSON file */
        storeFoldResult(foldResult,
                "latent_results/run_" + run + "/mixed/" + dataName + "/" + folds.size() + "_folds/GLSL_"+this.initialization,
                dataName + "_results_GLSL_"+this.initialization + "_"+  foldNumber  +".json");
    }

    private void storeFoldResult(Tuple4<BayesianNetwork, Double, Double, Long> foldResult,
                                 String directoryPath,
                                 String fileName) throws IOException {

        new File(directoryPath).mkdirs();

        LatentFoldResult jsonFoldResult = new LatentFoldResult();
        jsonFoldResult.setTest_LL(foldResult.getSecond());
        jsonFoldResult.setTrain_BIC(foldResult.getThird());
        jsonFoldResult.setLearning_time(foldResult.getFourth());

        /* Write the Json file */
        String output = directoryPath + "/" + fileName;
        try (Writer writer = new FileWriter(output)) {
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            gson.toJson(jsonFoldResult, writer);
        }
    }
}
