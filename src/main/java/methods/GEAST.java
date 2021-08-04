package methods;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.tuple.Tuple2;
import eu.amidst.extension.util.tuple.Tuple3;
import eu.amidst.extension.util.tuple.Tuple4;
import experiments.util.AmidstToLatlabData;
import experiments.util.EstimatePredictiveScore;
import org.latlab.core.data.MixedDataSet;
import org.latlab.core.learner.geast.GeastWithoutPouch;
import org.latlab.core.learner.geast.IModelWithScore;
import org.latlab.core.learner.geast.ParameterGenerator;
import org.latlab.core.learner.geast.Settings;
import org.latlab.core.model.Builder;
import org.latlab.core.model.Gltm;
import org.latlab.core.util.DiscreteVariable;
import org.latlab.core.util.Variable;

import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

// He modificado la clase para que vaya almzacenando los modelos intermedios porque tarda tanto entre cada fold que da problemas.
public class GEAST implements LatentContinuousMethod {

    private String settingsLocation;
    private long seed;
    private String dataNameAttribute;

    public GEAST(String settingsLocation,
                 long seed) {
        this.settingsLocation = settingsLocation;
        this.seed = seed;
    }

    public static Tuple4<IModelWithScore, Double, Double, Long> learnModel(DataOnMemory<DataInstance> trainData,
                                                                           String settingsLocation,
                                                                           long seed) throws Exception {

        System.out.println("\n==========================");
        System.out.println("GEAST");
        System.out.println("==========================\n");

        MixedDataSet trainDataLatlab = AmidstToLatlabData.transform(trainData);

        Settings settings = new Settings(settingsLocation, trainDataLatlab, trainDataLatlab.name());
        GeastWithoutPouch geastWithoutPouch = settings.createGeastWithoutPouch(seed);

        long initTime = System.currentTimeMillis();
        Gltm lcm =  Builder.buildNaiveBayesModel(
                new Gltm(),
                new DiscreteVariable(2),
                trainDataLatlab.getNonClassVariables());
        ParameterGenerator parameterGenerator = new ParameterGenerator(trainDataLatlab);
        parameterGenerator.generate(lcm);
        IModelWithScore result = geastWithoutPouch.learn(lcm);
        long endTime = System.currentTimeMillis();
        long learningTimeMs = (endTime - initTime);
        double learningTimeS = learningTimeMs / 1000;

        DecimalFormatSymbols otherSymbols = new DecimalFormatSymbols(Locale.getDefault());
        otherSymbols.setDecimalSeparator('.');
        System.out.println("\n---------------------------------------------");
        System.out.println("Log-Likelihood: " + result.loglikelihood());
        System.out.println("BIC: " + result.BicScore());
        System.out.println("Learning time (ms): " + learningTimeMs + " ms");
        System.out.println("Learning time (s): " + learningTimeS + " s");

        return new Tuple4<>(result, result.loglikelihood(), result.BicScore(), learningTimeMs);
    }

    @Override
    public void runLatentContinuous(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                    String dataName,
                                    int run,
                                    LogUtils.LogLevel foldLogLevel) throws Exception {

        this.dataNameAttribute = dataName;
        List<Gltm> models = new ArrayList<>(folds.size());
        List<Tuple3<Double, Double, Long>> scoresAndTimes = new ArrayList<>(folds.size());

        /* Run */
        List<Tuple4<Gltm, Double, Double, Long>> results = runLatent(folds, run, foldLogLevel);
        for(Tuple4<Gltm, Double, Double, Long> result: results) {
            models.add(result.getFirst());
            scoresAndTimes.add(new Tuple3<>(result.getSecond(), result.getThird(), result.getFourth()));
        }

        /* Store models */
//        storeContinuousModels(models, "latent_results/run_"+run+"/continuous/"+dataName+"/" + folds.size() +
//                "_folds/GEAST", dataName, "GEAST");

        /* Show average time and score */
        showAverageScoreAndTime(scoresAndTimes);

        /* Store experiment results in a JSON file */
        storeResults(scoresAndTimes, "latent_results/run_"+ run +"/continuous/"+ dataName+"/" + folds.size() +
                "_folds/GEAST", dataName + "_results_GEAST.json");
    }

    private List<Tuple4<Gltm, Double, Double, Long>> runLatent(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                                                               int run,
                                                               LogUtils.LogLevel foldLogLevel) throws Exception {

        System.out.println("\n==========================");
        System.out.println("GEAST");
        System.out.println("==========================\n");
        List<Tuple4<Gltm, Double, Double, Long>> foldsResults = new ArrayList<>();

        /* Iterate through the folds and learn an LCM on each one */
        for(int i = 0; i < folds.size(); i++) {
            Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>> fold = folds.get(i);

            DataOnMemory<DataInstance> trainData = fold.getFirst();
            DataOnMemory<DataInstance> testData = fold.getSecond();
            MixedDataSet trainDataLatlab = AmidstToLatlabData.transform(trainData);
            MixedDataSet testDataLatlab = prepareTestData(testData, trainDataLatlab.variables());

            Settings settings = new Settings(settingsLocation, trainDataLatlab, trainDataLatlab.name());
            GeastWithoutPouch geastWithoutPouch = settings.createGeastWithoutPouch(seed);

            long initTime = System.currentTimeMillis();
            Gltm lcm =  Builder.buildNaiveBayesModel(
                    new Gltm(),
                    new DiscreteVariable(2),
                    trainDataLatlab.getNonClassVariables());
            ParameterGenerator parameterGenerator = new ParameterGenerator(trainDataLatlab);
            parameterGenerator.setRandom(new Random(seed));
            parameterGenerator.generate(lcm);
            IModelWithScore result = geastWithoutPouch.learn(lcm);
            long endTime = System.currentTimeMillis();

            double testLL = EstimatePredictiveScore.latLabLL(result.model(), testDataLatlab);
            double trainBIC = EstimatePredictiveScore.latLabBIC(result.model(), trainDataLatlab);
            long foldTime = (endTime - initTime);
            foldsResults.add(new Tuple4<>(result.model(), testLL, trainBIC, foldTime));

            LogUtils.info("----------------------------------------", foldLogLevel);
            LogUtils.info("Fold " + (i+1) , foldLogLevel);
            LogUtils.info("Test Log-Likelihood: " + testLL, foldLogLevel);
            LogUtils.info("Train BIC: " + trainBIC, foldLogLevel);
            LogUtils.info("Time: " + foldTime + " ms", foldLogLevel);

            /* Store model */
//            storeContinuousModel(result.model(), i, "latent_results/run_"+run+"/continuous/"+dataNameAttribute+"/"
//                    + folds.size() + "_folds/GEAST", dataNameAttribute, "GEAST");
        }

        return foldsResults;
    }

    /** Both the test and train datasets need to have the same list of variable objects or it wont work (Latlab code) */
    private static MixedDataSet prepareTestData(DataOnMemory<DataInstance> amidstTestData, List<Variable> trainVariables) {
        MixedDataSet testData = MixedDataSet.createEmpty(trainVariables, amidstTestData.getNumberOfDataInstances());

        for(DataInstance instance: amidstTestData)
            testData.add(1, instance.toArray());

        return testData;
    }
}
