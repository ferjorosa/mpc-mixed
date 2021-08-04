package experiments.util;

import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.distribution.ConditionalDistribution;
import eu.amidst.core.distribution.Multinomial;
import eu.amidst.core.inference.ImportanceSampling;
import eu.amidst.core.inference.ImportanceSamplingRobust;
import eu.amidst.core.inference.InferenceEngine;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.core.models.DAG;
import eu.amidst.core.variables.*;
import org.latlab.core.data.MixedDataSet;
import org.latlab.core.learner.geast.DataPropagation;
import org.latlab.core.learner.geast.SharedTreePropagation;
import org.latlab.core.model.Gltm;
import org.latlab.core.reasoner.NaturalCliqueTreePropagation;
import voltric.data.DiscreteData;
import voltric.data.DiscreteDataInstance;
import voltric.learning.score.LearningScore;
import voltric.model.DiscreteBayesNet;
import voltric.util.Utils;

public class EstimatePredictiveScore {

    /** Transforma el modelo a Voltric y calcula su BIC */
    public static double discreteBIC(BayesianNetwork bayesianNetwork,
                                     DataOnMemory<DataInstance> data) {

        if(data.getAttributes().getFullListOfAttributes().stream().anyMatch(x->x.getStateSpaceType().getStateSpaceTypeEnum() == StateSpaceTypeEnum.REAL))
            throw new IllegalArgumentException("Continuous attributes not allowed");

        /* First we transform the Amidst network into Voltric format */
        DiscreteBayesNet voltricBLTM = AmidstToVoltricModel.transform(bayesianNetwork);

        /* Then we transform the Amidst data into Voltric format */
        DiscreteData voltricData = new DiscreteData(voltricBLTM.getManifestVariables());
        for(DataInstance instance: data){

            int[] instanceInt = new int[instance.getAttributes().getNumberOfAttributes()];
            for(int i = 0; i < instanceInt.length; i++)
                instanceInt[i] = (int) instance.toArray()[i];

            voltricData.add(new DiscreteDataInstance(instanceInt));
        }

        return voltricBIC(voltricBLTM, voltricData);

    }

    /** Transforma el modelo a Voltric y calcula su LL */
    public static double discreteLL(BayesianNetwork bayesianNetwork,
                                    DataOnMemory<DataInstance> data) {

        if(data.getAttributes().getFullListOfAttributes().stream().anyMatch(x->x.getStateSpaceType().getStateSpaceTypeEnum() == StateSpaceTypeEnum.REAL))
            throw new IllegalArgumentException("Continuous attributes not allowed");

        /* First we transform the Amidst network into Voltric format */
        DiscreteBayesNet voltricBLTM = AmidstToVoltricModel.transform(bayesianNetwork);

        /* Then we transform the Amidst data into Voltric format */
        DiscreteData voltricData = new DiscreteData(voltricBLTM.getManifestVariables());
        for(DataInstance instance: data){

            int[] instanceInt = new int[instance.getAttributes().getNumberOfAttributes()];
            for(int i = 0; i < instanceInt.length; i++)
                instanceInt[i] = (int) instance.toArray()[i];

            voltricData.add(new DiscreteDataInstance(instanceInt));
        }

        return voltricLL(voltricBLTM, voltricData);
    }

    public static double voltricBIC(DiscreteBayesNet bn,
                                    DiscreteData data) {
        return LearningScore.calculateBIC(data, bn);
    }

    public static double voltricLL(DiscreteBayesNet bn,
                                   DiscreteData data) {
        return LearningScore.calculateLogLikelihood(data, bn);
    }

    public static double amidstLL(BayesianNetwork bayesianNetwork, DataOnMemory<DataInstance> data) {

        ImportanceSampling importanceSampling = new ImportanceSampling();
        importanceSampling.setModel(bayesianNetwork);

        importanceSampling.setParallelMode(true);
        importanceSampling.setSampleSize(5000);
        importanceSampling.setSeed(0);
        importanceSampling.setKeepDataOnMemory(true);
        importanceSampling.runInference();

        double ll = 0;
        HashMapAssignment assignment = new HashMapAssignment(data.getAttributes().getNumberOfAttributes());
        for(DataInstance instance: data) {

            /* Prepare evidence */
            for(Attribute att: instance.getAttributes()) {
                Variable var = bayesianNetwork.getVariables().getVariableByName(att.getName());
                assignment.setValue(var, instance.getValue(att));
            }

            /* runInference and store ll score */
            importanceSampling.setEvidence(assignment);
            importanceSampling.runInference();
            double instanceLL = importanceSampling.getLogProbabilityOfEvidence();
            ll += instanceLL;
        }

        return ll;
    }

    public static double amidstAIC(BayesianNetwork bayesianNetwork, DataOnMemory<DataInstance> data) {
        int dim = bayesianNetwork.getNumberOfParameters();
        double ll = EstimatePredictiveScore.amidstLL(bayesianNetwork, data);
        return ll - dim;
    }

    public static double amidstBIC(BayesianNetwork bayesianNetwork, DataOnMemory<DataInstance> data) {
        int dim = bayesianNetwork.getNumberOfParameters();
        double ll = EstimatePredictiveScore.amidstLL(bayesianNetwork, data);
        return ll - (dim * Utils.log(data.getNumberOfDataInstances()) / 2.0);
    }

    public static double amidstLLRobust(BayesianNetwork bayesianNetwork, DataOnMemory<DataInstance> data) {

        ImportanceSamplingRobust importanceSampling = new ImportanceSamplingRobust();
        importanceSampling.setModel(bayesianNetwork);

        importanceSampling.setParallelMode(true);
        importanceSampling.setSampleSize(5000);
        importanceSampling.setSeed(0);
        importanceSampling.runInference();

        double ll = 0;
        HashMapAssignment assignment = new HashMapAssignment(data.getAttributes().getNumberOfAttributes());
        for(DataInstance instance: data) {

            /* Prepare evidence */
            for(Attribute att: instance.getAttributes()) {
                Variable var = bayesianNetwork.getVariables().getVariableByName(att.getName());
                assignment.setValue(var, instance.getValue(att));
            }

            /* runInference and store ll score */
            importanceSampling.setEvidence(assignment);
            importanceSampling.runInference();
            double instanceLL = importanceSampling.getLogProbabilityOfEvidence();
            ll += instanceLL;
        }

        return ll;
    }

    public static double latLabLL(Gltm model, MixedDataSet data) {
        DataPropagation testPropagation = new SharedTreePropagation(model, data);
        double testLL = 0;
        for (int i = 0; i < data.size(); i++) {
            NaturalCliqueTreePropagation ctp = testPropagation.compute(i);
            double weight = data.get(i).weight();
            testLL += ctp.loglikelihood() * weight;
        }
        return testLL;
    }

    public static double latLabBIC(Gltm model, MixedDataSet data) {
        DataPropagation testPropagation = new SharedTreePropagation(model, data);
        double testLL = 0;
        for (int i = 0; i < data.size(); i++) {
            NaturalCliqueTreePropagation ctp = testPropagation.compute(i);
            double weight = data.get(i).weight();
            testLL += ctp.loglikelihood() * weight;
        }
        return testLL - model.computeDimension() * Math.log(data.totalWeight()) / 2.0;
    }

    // TODO: Actualmente es con VMP y devuelve malos resultados
    // TODO: podriamos hacer que funcionase con redes de LCMs independientes, de tal forma que no hay que marginalizar con InferenceEngine
    @Deprecated
    public static double logLikelihoodAmidstVMP(BayesianNetwork bayesianNetwork,
                                                DataOnMemory<DataInstance> data) {

        Variables variables = bayesianNetwork.getVariables();
        DAG dag = bayesianNetwork.getDAG();

        double ll = 0;
        for(DataInstance dataInstance: data) {
            double instanceLL = 0;
            for(Attribute attribute: data.getAttributes()) {

                double instanceAttributeLL = 0;
                Variable var = variables.getVariableByName(attribute.getName());
                ConditionalDistribution dist = bayesianNetwork.getConditionalDistribution(var);

                if(dag.getParentSet(var).getNumberOfParents() == 1) {
                    Variable parent = dag.getParentSet(var).getParents().get(0);
                    Multinomial parentDist = InferenceEngine.getPosterior(parent, bayesianNetwork);
                    for(int i = 0; i < parent.getNumberOfStates(); i++){
                        Assignment assignment = new HashMapAssignment(2);
                        assignment.setValue(var, dataInstance.getValue(attribute));
                        assignment.setValue(parent, i);
                        instanceAttributeLL += parentDist.getProbabilityOfState(i) * dist.getLogConditionalProbability(assignment);
                    }
                }
                else {
                    Assignment assignment = new HashMapAssignment(1);
                    assignment.setValue(var, dataInstance.getValue(attribute));
                    instanceAttributeLL += dist.getLogConditionalProbability(assignment);
                }

                instanceLL += instanceAttributeLL;
            }
            ll += instanceLL;
        }
        return ll;
    }
}
