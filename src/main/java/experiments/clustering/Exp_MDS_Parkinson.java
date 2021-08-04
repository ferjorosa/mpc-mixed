package experiments.clustering;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.io.BayesianNetworkWriter;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.core.io.DataStreamWriter;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.Variables;
import eu.amidst.extension.data.DataUtils;
import eu.amidst.extension.io.GenieWriter;
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
import eu.amidst.extension.missing.util.ImputeMissing;
import eu.amidst.extension.util.EstimatePredictiveScore;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.PriorsFromData;
import eu.amidst.extension.util.tuple.Tuple2;
import eu.amidst.extension.util.tuple.Tuple3;
import experiments.util.JsonResult;
import methods.CIL;
import methods.IL;
import methods.UKDB;
import methods.VariationalLCM;

import java.io.File;
import java.io.FileWriter;
import java.io.Writer;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;

public class Exp_MDS_Parkinson {

    public static void main(String[] args) throws Exception {

        long seed = 0;
        String resultsPath = "clustering_results/mds_parkinson/";
        int nVbsemCandidates = 1;
        int nVbemCandidates = 64;

        learnLCM(seed, resultsPath, nVbemCandidates, LogUtils.LogLevel.INFO);
        learnUKDB(seed, resultsPath, nVbemCandidates, nVbsemCandidates, LogUtils.LogLevel.INFO, LogUtils.LogLevel.NONE, LogUtils.LogLevel.NONE);
        learnCIL(seed, resultsPath, nVbemCandidates, 1, LogUtils.LogLevel.INFO);
        learnCIL(seed, resultsPath, nVbemCandidates, 10, LogUtils.LogLevel.INFO);
        learnIL(seed, resultsPath, nVbemCandidates, LogUtils.LogLevel.INFO);
        learnGLSL(seed, resultsPath, nVbemCandidates, nVbsemCandidates, Integer.MAX_VALUE, 1, LogUtils.LogLevel.INFO);
        learnGLSL_CIL(seed, resultsPath, nVbemCandidates, nVbsemCandidates, 1, Integer.MAX_VALUE, 1, LogUtils.LogLevel.INFO);
        learnGLSL_CIL(seed, resultsPath, nVbemCandidates, nVbsemCandidates, 10, Integer.MAX_VALUE, 1, LogUtils.LogLevel.INFO);
    }

    private static void learnLCM(long seed,
                                 String resultsPath,
                                 int nVbemCandidates,
                                 LogUtils.LogLevel logLevel) throws Exception {

        new File(resultsPath).mkdirs();

        String fileName = "mds_parkinson_LCM";

        DataOnMemory<DataInstance> data = DataStreamLoader.loadDataOnMemoryFromFile("data/mds_parkinson/mds_parkinson_train.arff");

        /* Generate Empirical Bayes priors from data, ignoring missing values */
        Map<String, double[]> priors = PriorsFromData.generate(data, 1);

        Tuple3<BayesianNetwork, Double, Long> result = VariationalLCM.learnModel(data, seed, priors, logLevel, false);
        BayesianNetwork posteriorPredictive = result.getFirst();
        double elbo = result.getSecond();
        long learningTime = result.getThird();
        double logLikelihood = EstimatePredictiveScore.amidstLL(posteriorPredictive, data);
        double bic = EstimatePredictiveScore.amidstBIC(posteriorPredictive, data);
        double aic = EstimatePredictiveScore.amidstAIC(posteriorPredictive, data);

        System.out.println("---------------------");
        System.out.println("Learning time (seconds): " + learningTime);
        System.out.println("ELBO: " + elbo);
        System.out.println("LogLikelihood: " + logLikelihood);
        System.out.println("BIC: " + bic);
        System.out.println("AIC: " + aic);

        int nParams = posteriorPredictive.getNumberOfParameters();
        int nClusteringVars = (int) posteriorPredictive.getVariables().getListOfVariables().stream().filter(x->x.isDiscrete() && !x.isObservable()).count();

        System.out.println("Num of params: " + nParams);
        System.out.println("Num of clustering vars: " + nClusteringVars);

        /* Write the Json result file */
        JsonResult jsonResult = new JsonResult(learningTime, elbo, logLikelihood, bic, aic,0, nVbemCandidates, nParams, nClusteringVars);
        try (Writer writer = new FileWriter(resultsPath + "/" + fileName + ".json")) {
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            gson.toJson(jsonResult, writer);
        }

        /*
         * Complete latent variables
         */
        DataOnMemory<DataInstance> completeDataWithLatents = DataUtils.completeLatentData(data, posteriorPredictive);
        DataStreamWriter.writeDataToFile(completeDataWithLatents, resultsPath + "/" + fileName + ".arff");

        /* Write the XDSL (Genie) file */
        DataUtils.defineAttributesMaxMinValues(completeDataWithLatents);
        GenieWriter genieWriter = new GenieWriter();
        genieWriter.write(posteriorPredictive, new File(resultsPath + "/" + fileName + ".xdsl"));

        /* Export model in AMIDST format */
        BayesianNetworkWriter.save(posteriorPredictive, resultsPath + "/" + fileName + ".bn");
    }

    private static void learnUKDB(long seed,
                                  String resultsPath,
                                  int nVbemCandidates,
                                  int nVbsemCandidates,
                                  LogUtils.LogLevel kdbLogLevel,
                                  LogUtils.LogLevel vbsemLogLevel,
                                  LogUtils.LogLevel hcLogLevel) throws Exception {

        new File(resultsPath).mkdirs();

        String fileName = "mds_parkinson_UKDB";

        System.out.println("==================================");
        System.out.println("=============== UKDB =============");
        System.out.println("==================================");
        System.out.println("n VBEM candidates: " + nVbemCandidates);
        System.out.println("n VBSEM candidates: " + nVbsemCandidates);

        DataOnMemory<DataInstance> data = DataStreamLoader.loadDataOnMemoryFromFile("data/mds_parkinson/mds_parkinson_train.arff");

        /* Generate Empirical Bayes priors from data, ignoring missing values */
        Map<String, double[]> priors = PriorsFromData.generate(data, 1);

        InitializationVBEM initializationVBEM = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, nVbemCandidates, nVbemCandidates/2, false);
        VBEMConfig vbemConfig = new VBEMConfig(seed, 0.01, 100, initializationVBEM, new BishopPenalizer());
        BayesianHcConfig bayesianHcConfig = new BayesianHcConfig(seed, 0.01, 100);
        InitializationVBSEM initializationVBSEM = new InitializationVBSEM(InitializationTypeVBSEM.NONE, nVbsemCandidates, nVbsemCandidates/2, 0.2, true);
        UKDB UKDBMethod = new UKDB(vbemConfig, bayesianHcConfig, 100, initializationVBSEM);

        long initTime = System.currentTimeMillis();
        Tuple2<BayesianNetwork, Double> result = UKDBMethod.learnModelIteratively(data, 2, 100, priors, kdbLogLevel, vbsemLogLevel, hcLogLevel);
        long endTime = System.currentTimeMillis();

        BayesianNetwork posteriorPredictive = result.getFirst();
        double elbo = result.getSecond();
        double logLikelihood = EstimatePredictiveScore.amidstLL(posteriorPredictive, data);
        double bic = EstimatePredictiveScore.amidstBIC(posteriorPredictive, data);
        double aic = EstimatePredictiveScore.amidstAIC(posteriorPredictive, data);
        double learningTime = (endTime - initTime) / 1000.0;

        System.out.println("---------------------");
        System.out.println("Learning time (seconds): " + learningTime);
        System.out.println("ELBO: " + elbo);
        System.out.println("LogLikelihood: " + logLikelihood);
        System.out.println("BIC: " + bic);
        System.out.println("AIC: " + aic);

        int nParams = posteriorPredictive.getNumberOfParameters();
        int nClusteringVars = (int) posteriorPredictive.getVariables().getListOfVariables().stream().filter(x->x.isDiscrete() && !x.isObservable()).count();

        System.out.println("Num of params: " + nParams);
        System.out.println("Num of clustering vars: " + nClusteringVars);

        /* Write the Json result file */
        JsonResult jsonResult = new JsonResult(learningTime, elbo, logLikelihood, bic, aic,0, nVbemCandidates, nParams, nClusteringVars);
        try (Writer writer = new FileWriter(resultsPath + "/" + fileName + ".json")) {
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            gson.toJson(jsonResult, writer);
        }

        /*
         * Complete latent variables
         */
        DataOnMemory<DataInstance> completeDataWithLatents = DataUtils.completeLatentData(data, posteriorPredictive);
        DataStreamWriter.writeDataToFile(completeDataWithLatents, resultsPath + "/" + fileName + ".arff");

        /* Write the XDSL (Genie) file */
        DataUtils.defineAttributesMaxMinValues(completeDataWithLatents);
        GenieWriter genieWriter = new GenieWriter();
        genieWriter.write(posteriorPredictive, new File(resultsPath + "/" + fileName + ".xdsl"));

        /* Export model in AMIDST format */
        BayesianNetworkWriter.save(posteriorPredictive, resultsPath + "/" + fileName + ".bn");
    }

    private static void learnIL(long seed,
                                String resultsPath,
                                int nVbemCandidates,
                                LogUtils.LogLevel logLevel) throws Exception {

        new File(resultsPath).mkdirs();

        String fileName = "mds_parkinson_IL";

        DataOnMemory<DataInstance> data = DataStreamLoader.loadDataOnMemoryFromFile("data/mds_parkinson/mds_parkinson_train.arff");

        /* Generate Empirical Bayes priors from data, ignoring missing values */
        Map<String, double[]> priors = PriorsFromData.generate(data, 1);

        Tuple3<BayesianNetwork, Double, Long> result = IL.learnModel(data, priors, seed, true, true, false, new SimpleLocalVBEM(), logLevel, false);

        BayesianNetwork posteriorPredictive = result.getFirst();
        double elbo = result.getSecond();
        long learningTime = result.getThird();
        double logLikelihood = EstimatePredictiveScore.amidstLL(posteriorPredictive, data);
        double bic = EstimatePredictiveScore.amidstBIC(posteriorPredictive, data);
        double aic = EstimatePredictiveScore.amidstAIC(posteriorPredictive, data);

        System.out.println("---------------------");
        System.out.println("Learning time (seconds): " + learningTime);
        System.out.println("ELBO: " + elbo);
        System.out.println("LogLikelihood: " + logLikelihood);
        System.out.println("BIC: " + bic);
        System.out.println("AIC: " + aic);

        int nParams = posteriorPredictive.getNumberOfParameters();
        int nClusteringVars = (int) posteriorPredictive.getVariables().getListOfVariables().stream().filter(x->x.isDiscrete() && !x.isObservable()).count();

        System.out.println("Num of params: " + nParams);
        System.out.println("Num of clustering vars: " + nClusteringVars);

        /* Write the Json result file */
        JsonResult jsonResult = new JsonResult(learningTime, elbo, logLikelihood, bic, aic,0, nVbemCandidates, nParams, nClusteringVars);
        try (Writer writer = new FileWriter(resultsPath + "/" + fileName + ".json")) {
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            gson.toJson(jsonResult, writer);
        }

        /*
         * Complete latent variables
         */
        DataOnMemory<DataInstance> completeDataWithLatents = DataUtils.completeLatentData(data, posteriorPredictive);
        DataStreamWriter.writeDataToFile(completeDataWithLatents, resultsPath + "/" + fileName + ".arff");

        /* Write the XDSL (Genie) file */
        DataUtils.defineAttributesMaxMinValues(completeDataWithLatents);
        GenieWriter genieWriter = new GenieWriter();
        genieWriter.write(posteriorPredictive, new File(resultsPath + "/" + fileName + ".xdsl"));

        /* Export model in AMIDST format */
        BayesianNetworkWriter.save(posteriorPredictive, resultsPath + "/" + fileName + ".bn");
    }

    private static void learnCIL(long seed,
                                 String resultsPath,
                                 int nVbemCandidates,
                                 int alpha,
                                 LogUtils.LogLevel logLevel) throws Exception {

        new File(resultsPath).mkdirs();

        String fileName = "mds_parkinson_CIL_" + alpha;

        DataOnMemory<DataInstance> data = DataStreamLoader.loadDataOnMemoryFromFile("data/mds_parkinson/mds_parkinson_train.arff");

        /* Generate Empirical Bayes priors from data, ignoring missing values */
        Map<String, double[]> priors = PriorsFromData.generate(data, 1);

        Tuple3<BayesianNetwork, Double, Long> result = CIL.learnModel(data, priors, seed, alpha, true, true, false, 3, false, false, new SimpleLocalVBEM(), logLevel, false);

        BayesianNetwork posteriorPredictive = result.getFirst();
        double elbo = result.getSecond();
        long learningTime = result.getThird();
        double logLikelihood = EstimatePredictiveScore.amidstLL(posteriorPredictive, data);
        double bic = EstimatePredictiveScore.amidstBIC(posteriorPredictive, data);
        double aic = EstimatePredictiveScore.amidstAIC(posteriorPredictive, data);

        System.out.println("---------------------");
        System.out.println("Learning time (seconds): " + learningTime);
        System.out.println("ELBO: " + elbo);
        System.out.println("LogLikelihood: " + logLikelihood);
        System.out.println("BIC: " + bic);
        System.out.println("AIC: " + aic);

        int nParams = posteriorPredictive.getNumberOfParameters();
        int nClusteringVars = (int) posteriorPredictive.getVariables().getListOfVariables().stream().filter(x->x.isDiscrete() && !x.isObservable()).count();

        System.out.println("Num of params: " + nParams);
        System.out.println("Num of clustering vars: " + nClusteringVars);

        /* Write the Json result file */
        JsonResult jsonResult = new JsonResult(learningTime, elbo, logLikelihood, bic, aic,0, nVbemCandidates, nParams, nClusteringVars);
        try (Writer writer = new FileWriter(resultsPath + "/" + fileName + ".json")) {
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            gson.toJson(jsonResult, writer);
        }

        /*
         * Complete latent variables
         */
        DataOnMemory<DataInstance> completeDataWithLatents = DataUtils.completeLatentData(data, posteriorPredictive);
        DataStreamWriter.writeDataToFile(completeDataWithLatents, resultsPath + "/" + fileName + ".arff");

        /* Write the XDSL (Genie) file */
        DataUtils.defineAttributesMaxMinValues(completeDataWithLatents);
        GenieWriter genieWriter = new GenieWriter();
        genieWriter.write(posteriorPredictive, new File(resultsPath + "/" + fileName + ".xdsl"));

        /* Export model in AMIDST format */
        BayesianNetworkWriter.save(posteriorPredictive, resultsPath + "/" + fileName + ".bn");
    }

    private static void learnGLSL(long seed,
                                  String resultsPath,
                                  int nVbemCandidates,
                                  int nVbsemCandidates,
                                  int maxNumberParents_latent,
                                  int maxNumberParents_observed,
                                  LogUtils.LogLevel logLevel) throws Exception{

        new File(resultsPath).mkdirs();

        String fileName = "mds_parkinson_GLSL";

        System.out.println("==================================");
        System.out.println("============== GLSL ==============");
        System.out.println("==================================");
        System.out.println("n VBEM candidates: " + nVbemCandidates);
        System.out.println("n VBSEM candidates: " + nVbsemCandidates);

        DataOnMemory<DataInstance> data = DataStreamLoader.loadDataOnMemoryFromFile("data/mds_parkinson/mds_parkinson_train.arff");

        /* Generate Empirical Bayes priors from data, ignoring missing values */
        Map<String, double[]> priors = PriorsFromData.generate(data, 1);

        /**************************************************************************************************************/
        /* Generate the empty network (all variables are independent and there are no latent vars) and learn its parameters */
        Variables variables = new Variables(data.getAttributes());
        DAG emptyDag = new DAG(variables);
        VBEM vbem = new VBEM();
        double emptyDagScore = vbem.learnModelWithPriorUpdate(data, emptyDag);
        BayesianNetwork emptyBn = vbem.getLearntBayesianNetwork();
        Tuple2<BayesianNetwork, Double> resultEmpty = new Tuple2<>(emptyBn, emptyDagScore);

        /**************************************************************************************************************/

        InitializationVBEM initializationVBEM = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, nVbemCandidates, 16, true);
        VBEMConfig vbemConfig = new VBEMConfig(seed, 0.01, 100, initializationVBEM, new BishopPenalizer());
        InitializationVBSEM initializationVBSEM = new InitializationVBSEM(InitializationTypeVBSEM.NONE, nVbsemCandidates, 16, 0.2, true);
        BayesianHcConfig bayesianHcConfig = new BayesianHcConfig(seed, 0.01, 100);

        LatentVarCounter latentVarCounter = new LatentVarCounter();
        GLSL_IncreaseCard glsl_increaseCard = new GLSL_IncreaseCard(Integer.MAX_VALUE, maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM);
        GLSL_DecreaseCard glsl_decreaseCard = new GLSL_DecreaseCard(2, maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM);
        GLSL_RemoveLatent glsl_removeLatent = new GLSL_RemoveLatent(maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM);
        GLSL_AddLatent_child glsl_addLatent_child = new GLSL_AddLatent_child(Integer.MAX_VALUE, maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM, latentVarCounter);
        GLSL_AddLatent_ind glsl_addLatent_ind = new GLSL_AddLatent_ind(Integer.MAX_VALUE, 2, maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM, latentVarCounter);
        Set<GLSL_Operator> glslOperators = new LinkedHashSet<>(2);
        glslOperators.add(glsl_increaseCard);
        glslOperators.add(glsl_decreaseCard);
        glslOperators.add(glsl_removeLatent);
        glslOperators.add(glsl_addLatent_child);
        glslOperators.add(glsl_addLatent_ind);

        long initTime = System.currentTimeMillis();
        GLSL glsl = new GLSL(Integer.MAX_VALUE, glslOperators);
        Tuple2<BayesianNetwork, Double> result = glsl.learnModel(resultEmpty.getFirst(), resultEmpty.getSecond(), data, logLevel, logLevel);
        long endTime = System.currentTimeMillis();

        BayesianNetwork posteriorPredictive = result.getFirst();
        double elbo = result.getSecond();
        double logLikelihood = EstimatePredictiveScore.amidstLL(posteriorPredictive, data);
        double bic = EstimatePredictiveScore.amidstBIC(posteriorPredictive, data);
        double aic = EstimatePredictiveScore.amidstAIC(posteriorPredictive, data);
        double learningTime = (endTime - initTime) / 1000.0;

        System.out.println("---------------------");
        System.out.println("Learning time (seconds): " + learningTime);
        System.out.println("ELBO: " + elbo);
        System.out.println("LogLikelihood: " + logLikelihood);
        System.out.println("BIC: " + bic);
        System.out.println("AIC: " + aic);

        int nParams = posteriorPredictive.getNumberOfParameters();
        int nClusteringVars = (int) posteriorPredictive.getVariables().getListOfVariables().stream().filter(x->x.isDiscrete() && !x.isObservable()).count();

        System.out.println("Num of params: " + nParams);
        System.out.println("Num of clustering vars: " + nClusteringVars);

        /* Write the Json result file */
        JsonResult jsonResult = new JsonResult(learningTime, elbo, logLikelihood, bic, aic,0, nVbemCandidates, nParams, nClusteringVars);
        try (Writer writer = new FileWriter(resultsPath + "/" + fileName + ".json")) {
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            gson.toJson(jsonResult, writer);
        }

        /*
         * Impute missing values with the posterior predictive, then impute latent variables values for each data instance
         * and write the resulting dataset with imputed missing and new completed latent vars
         */
        DataOnMemory<DataInstance> completeData = ImputeMissing.imputeWithModel(data, posteriorPredictive);
        DataOnMemory<DataInstance> completeDataWithLatents = DataUtils.completeLatentData(completeData, posteriorPredictive);
        DataStreamWriter.writeDataToFile(completeDataWithLatents, resultsPath + "/" + fileName + ".arff");

        /* Write the XDSL (Genie) file */
        DataUtils.defineAttributesMaxMinValues(completeDataWithLatents);
        GenieWriter genieWriter = new GenieWriter();
        genieWriter.write(posteriorPredictive, new File(resultsPath + "/" + fileName + ".xdsl"));

        /* Export model in AMIDST format */
        BayesianNetworkWriter.save(posteriorPredictive, resultsPath + "/" + fileName + ".bn");
    }

    private static void learnGLSL_CIL(long seed,
                                  String resultsPath,
                                  int nVbemCandidates,
                                  int nVbsemCandidates,
                                  int alpha,
                                  int maxNumberParents_latent,
                                  int maxNumberParents_observed,
                                  LogUtils.LogLevel logLevel) throws Exception{

        new File(resultsPath).mkdirs();

        String fileName = "mds_parkinson_GLSL_CIL_" + alpha;

        System.out.println("==================================");
        System.out.println("========== GLSL with CIL ==========");
        System.out.println("==================================");
        System.out.println("n VBEM candidates: " + nVbemCandidates);
        System.out.println("n VBSEM candidates: " + nVbsemCandidates);
        System.out.println("CIL alpha: " + alpha);

        DataOnMemory<DataInstance> data = DataStreamLoader.loadDataOnMemoryFromFile("data/mds_parkinson/mds_parkinson_train.arff");

        /* Generate Empirical Bayes priors from data, ignoring missing values */
        Map<String, double[]> priors = PriorsFromData.generate(data, 1);

        /**************************************************************************************************************/
        /**************************************************************************************************************/

        long initTime = System.currentTimeMillis();

        /* Run the CIL method */
        long initTimeCIL = System.currentTimeMillis();
        Tuple3<BayesianNetwork, Double, Long> resultCIL = CIL.learnModel(data, priors, seed, alpha, true, true, false, 3, false, false, new SimpleLocalVBEM(), logLevel, false);
        long endTimeCIL = System.currentTimeMillis();

        long learningTimeCIL = endTimeCIL - initTimeCIL;

        System.out.println("Time CIL: " + learningTimeCIL);

        /**************************************************************************************************************/

        InitializationVBEM initializationVBEM = new InitializationVBEM(InitializationTypeVBEM.PYRAMID, nVbemCandidates, 16, true);
        VBEMConfig vbemConfig = new VBEMConfig(seed, 0.01, 100, initializationVBEM, new BishopPenalizer());
        InitializationVBSEM initializationVBSEM = new InitializationVBSEM(InitializationTypeVBSEM.NONE, nVbsemCandidates, 16, 0.2, true);
        BayesianHcConfig bayesianHcConfig = new BayesianHcConfig(seed, 0.01, 100);

        LatentVarCounter latentVarCounter = new LatentVarCounter();
        GLSL_IncreaseCard glsl_increaseCard = new GLSL_IncreaseCard(Integer.MAX_VALUE, maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM);
        GLSL_DecreaseCard glsl_decreaseCard = new GLSL_DecreaseCard(2, maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM);
        GLSL_RemoveLatent glsl_removeLatent = new GLSL_RemoveLatent(maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM);
        GLSL_AddLatent_child glsl_addLatent_child = new GLSL_AddLatent_child(Integer.MAX_VALUE, maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM, latentVarCounter);
        GLSL_AddLatent_ind glsl_addLatent_ind = new GLSL_AddLatent_ind(Integer.MAX_VALUE, 2, maxNumberParents_latent, maxNumberParents_observed, vbemConfig, bayesianHcConfig, initializationVBSEM, latentVarCounter);
        Set<GLSL_Operator> glslOperators = new LinkedHashSet<>(2);
        glslOperators.add(glsl_increaseCard);
        glslOperators.add(glsl_decreaseCard);
        glslOperators.add(glsl_removeLatent);
        glslOperators.add(glsl_addLatent_child);
        glslOperators.add(glsl_addLatent_ind);


        GLSL glsl = new GLSL(Integer.MAX_VALUE, glslOperators);
        Tuple2<BayesianNetwork, Double> result = glsl.learnModel(resultCIL.getFirst(), resultCIL.getSecond(), data, logLevel, logLevel);
        long endTime = System.currentTimeMillis();

        BayesianNetwork posteriorPredictive = result.getFirst();
        double elbo = result.getSecond();
        double logLikelihood = EstimatePredictiveScore.amidstLL(posteriorPredictive, data);
        double bic = EstimatePredictiveScore.amidstBIC(posteriorPredictive, data);
        double aic = EstimatePredictiveScore.amidstAIC(posteriorPredictive, data);
        double learningTime = (endTime - initTime) / 1000.0;

        System.out.println("---------------------");
        System.out.println("Learning time (seconds): " + learningTime);
        System.out.println("ELBO: " + elbo);
        System.out.println("LogLikelihood: " + logLikelihood);
        System.out.println("BIC: " + bic);
        System.out.println("AIC: " + aic);

        int nParams = posteriorPredictive.getNumberOfParameters();
        int nClusteringVars = (int) posteriorPredictive.getVariables().getListOfVariables().stream().filter(x->x.isDiscrete() && !x.isObservable()).count();

        System.out.println("Num of params: " + nParams);
        System.out.println("Num of clustering vars: " + nClusteringVars);

        /* Write the Json result file */
        JsonResult jsonResult = new JsonResult(learningTime, elbo, logLikelihood, bic, aic,0, nVbemCandidates, nParams, nClusteringVars);
        try (Writer writer = new FileWriter(resultsPath + "/" + fileName + ".json")) {
            Gson gson = new GsonBuilder().setPrettyPrinting().create();
            gson.toJson(jsonResult, writer);
        }

        /*
         * Impute missing values with the posterior predictive, then impute latent variables values for each data instance
         * and write the resulting dataset with imputed missing and new completed latent vars
         */
        DataOnMemory<DataInstance> completeData = ImputeMissing.imputeWithModel(data, posteriorPredictive);
        DataOnMemory<DataInstance> completeDataWithLatents = DataUtils.completeLatentData(completeData, posteriorPredictive);
        DataStreamWriter.writeDataToFile(completeDataWithLatents, resultsPath + "/" + fileName + ".arff");

        /* Write the XDSL (Genie) file */
        DataUtils.defineAttributesMaxMinValues(completeDataWithLatents);
        GenieWriter genieWriter = new GenieWriter();
        genieWriter.write(posteriorPredictive, new File(resultsPath + "/" + fileName + ".xdsl"));

        /* Export model in AMIDST format */
        BayesianNetworkWriter.save(posteriorPredictive, resultsPath + "/" + fileName + ".bn");
    }
}
