package experiments.latent.discrete;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.extension.learn.structure.typelocalvbem.SimpleLocalVBEM;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.PriorsFromData;
import eu.amidst.extension.util.tuple.Tuple2;
import experiments.latent.LatentCrossValidationExperiment;
import experiments.latent.LatentDiscreteExperiment;
import experiments.util.GenerateLatentData;
import methods.*;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class Exp_alarm extends LatentDiscreteExperiment implements LatentCrossValidationExperiment {

    public Exp_alarm(Set<LatentDiscreteMethod> methods) {
        super(methods);
    }

    public static void main(String[] args) throws Exception {

        long seed = 0;
        int kFolds = 10;
        int run = 1;
        LogUtils.LogLevel logLevel = LogUtils.LogLevel.INFO;

        Set<LatentDiscreteMethod> methods = new LinkedHashSet<>();
        methods.add(new HC());
        methods.add(new IL(seed, false, true, true, new SimpleLocalVBEM()));
        methods.add(new GLSL_Algorithm(seed, 3, 3, Integer.MAX_VALUE, 64, 1, GLSL_Algorithm.Initialization.EMPTY));

        Exp_alarm exp = new Exp_alarm(methods);
        exp.runCrossValExperiment(seed, kFolds, run, logLevel);
    }

    @Override
    public void runCrossValExperiment(long seed, int kFolds, int run, LogUtils.LogLevel foldLogLevel) throws Exception {

        System.out.println("------------------------------------------------------------------");
        System.out.println("------------------------------------------------------------------");
        System.out.println("------------------------------ ALARM -----------------------------");
        System.out.println("------------------------------------------------------------------");
        System.out.println("------------------------------------------------------------------");

        String filename = "data/discrete/alarm.arff";
        String dataName = "alarm";
        DataOnMemory<DataInstance> data = DataStreamLoader.open(filename).toDataOnMemory();


        List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds = GenerateLatentData.generate(data, kFolds);
        System.out.println(kFolds + " folds have been generated");

        /* Filter Bayesian methods and assign them the empirical Bayes priors */
        final Map<String, double[]> priors = PriorsFromData.generate(data, 1);
        methods.stream()
                .filter(x -> x instanceof BayesianMethod)
                .forEach(x-> ((BayesianMethod) x).setPriors(priors));

        /* Run methods */
        for(LatentDiscreteMethod method: methods)
            method.runLatentDiscrete(folds, "alarm", run, LogUtils.LogLevel.INFO);
    }
}
