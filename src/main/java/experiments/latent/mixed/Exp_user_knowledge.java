package experiments.latent.mixed;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.extension.data.DataUtils;
import eu.amidst.extension.learn.structure.typelocalvbem.SimpleLocalVBEM;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.PriorsFromData;
import eu.amidst.extension.util.tuple.Tuple2;
import experiments.latent.LatentCrossValidationExperiment;
import experiments.latent.LatentMixedExperiment;
import experiments.util.GenerateLatentData;
import methods.*;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Exp_user_knowledge extends LatentMixedExperiment implements LatentCrossValidationExperiment {

    public Exp_user_knowledge(Set<LatentMixedMethod> methods) { super(methods); }

    public static void main(String[] args) throws Exception {
        long seed = 0;
        int kFolds = 10;
        int run = 1;
        LogUtils.LogLevel logLevel = LogUtils.LogLevel.INFO;

        Set<LatentMixedMethod> methods = new LinkedHashSet<>();
        methods.add(new HC());
        methods.add(new IL(seed, false, true, true, new SimpleLocalVBEM()));
        methods.add(new GLSL_Algorithm(seed, 3, 3, Integer.MAX_VALUE, 64, 1, GLSL_Algorithm.Initialization.EMPTY));

        Exp_user_knowledge experiment = new Exp_user_knowledge(methods);
        experiment.runCrossValExperiment(seed, kFolds, run, logLevel);
    }

    @Override
    public void runCrossValExperiment(long seed, int kFolds, int run, LogUtils.LogLevel foldLogLevel) throws Exception {
        System.out.println("------------------------------------------------------------------");
        System.out.println("------------------------------------------------------------------");
        System.out.println("-------------------------- USER_KNOWLEDGE ------------------------");
        System.out.println("------------------------------------------------------------------");
        System.out.println("------------------------------------------------------------------");

        String dataName = "user_knowledge";
        String filename = "data/mixed/"+dataName+".arff";
        DataOnMemory<DataInstance> data = DataStreamLoader.open(filename).toDataOnMemory();
        DataUtils.defineAttributesMaxMinValues(data);

        /* Generate data folds */
        List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds = GenerateLatentData.generate(data, kFolds);
        System.out.println(kFolds + " folds have been generated");

        /* Filter Bayesian methods and assign them the empirical Bayes priors */
        final Map<String, double[]> priors = PriorsFromData.generate(data, 1);
        this.methods.stream()
                .filter(x -> x instanceof BayesianMethod)
                .forEach(x -> ((BayesianMethod) x).setPriors(priors));

        /* Run methods */
        for (LatentMixedMethod method : methods)
            method.runLatentMixed(folds, dataName, run, LogUtils.LogLevel.INFO);
    }
}

