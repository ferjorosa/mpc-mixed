package voltric.learning.structure.incremental;

import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.tuple.Tuple3;
import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.EM;
import voltric.learning.parameter.em.config.EmConfig;
import voltric.learning.structure.incremental.localemtype.TypeLocalEM;
import voltric.learning.structure.incremental.operator.LfmIncOperator;
import voltric.learning.structure.incremental.operator.cardinality.LfmIncDecreaseCard;
import voltric.learning.structure.incremental.operator.cardinality.LfmIncIncreaseCard;
import voltric.model.DiscreteBayesNet;
import voltric.variables.DiscreteVariable;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

// TODO: Uso internamente String como identificador en currentSet, pero su hashcode es malo, seria mejor usar un objeto "Variable" similar al de Amidst donde se no se considerase el StateSpace
public class LFM_IncLearnerMax {

    private Set<LfmIncOperator> operators;

    private boolean iterationGlobalEM;

    private EmConfig initialEMConfig;

    private EmConfig localEMConfig;

    private EmConfig iterationEMConfig;

    private EmConfig finalEMConfig;

    private TypeLocalEM typeLocalEM;

    public LFM_IncLearnerMax(Set<LfmIncOperator> operators,
                             boolean iterationGlobalEM,
                             EmConfig initialEMConfig,
                             EmConfig localEMConfig,
                             EmConfig iterationEMConfig,
                             EmConfig finalEMConfig,
                             TypeLocalEM typeLocalEM) {
        this.operators = operators;
        this.initialEMConfig = initialEMConfig;
        this.localEMConfig = localEMConfig;
        this.iterationEMConfig = iterationEMConfig;
        this.finalEMConfig = finalEMConfig;
        this.iterationGlobalEM = iterationGlobalEM;
        this.typeLocalEM = typeLocalEM;
    }


    public LearningResult<DiscreteBayesNet> learnModel(DiscreteData data,
                                                       LogUtils.LogLevel logLevel) {

        /* Partimos de un modelo inicial donde todas son observadas e independientes */
        DiscreteBayesNet initialModel = new DiscreteBayesNet();
        for(DiscreteVariable variable: data.getVariables())
            initialModel.addNode(variable);

        /* Set de variables consideradas */
        List<String> currentSet = new ArrayList<>(initialModel.getVariables().size()); // Current set of variables being considered
        for(DiscreteVariable var: initialModel.getVariables())
            currentSet.add(var.getName());

        /* Aprendemos sus parametros */
        EM initialEM = new EM(this.initialEMConfig);
        LearningResult<DiscreteBayesNet> bestResult = initialEM.learnModel(initialModel, data);

        LogUtils.info("Initial score: " + bestResult.getScoreValue(), logLevel);

        /* 1 - Main loop */
        boolean keepsImproving = true;
        int iteration = 0;
        while(keepsImproving && currentSet.size() > 1) {

            iteration++;
            LearningResult<DiscreteBayesNet> bestIterationResult =
                    new LearningResult<>(null, -Double.MAX_VALUE, this.localEMConfig.getScoreType());
            Tuple3<DiscreteVariable, DiscreteVariable, LearningResult<DiscreteBayesNet>> bestIterationTriple =
                    new Tuple3<>(null, null, bestIterationResult);

            /* 1.1 - Iterate through the operators and select the one that returns the best model */
            for(LfmIncOperator operator: operators) {
                Tuple3<DiscreteVariable, DiscreteVariable, LearningResult<DiscreteBayesNet>> operatorTriple =
                        operator.apply(currentSet, bestResult.getBayesianNetwork(), data);

                double operatorScore = operatorTriple.getThird().getScoreValue();
                if(operatorScore == -Double.MAX_VALUE)
                    LogUtils.debug(operatorTriple.getThird().getName() + " -> NONE", logLevel);
                else
                    LogUtils.debug(operatorTriple.getThird().getName() + "(" + operatorTriple.getFirst().getName()+"," + operatorTriple.getSecond().getName()+") -> " + operatorTriple.getThird().getScoreValue(), logLevel);

                if(operatorScore > bestIterationTriple.getThird().getScoreValue()) {
                    bestIterationTriple = operatorTriple;
                    bestIterationResult = bestIterationTriple.getThird();
                }
            }

            /* 1.2 - Select the latent variables in the pair */
            List<String> latentVariables = new ArrayList<>();
            DiscreteVariable firstVar = bestIterationTriple.getFirst();
            DiscreteVariable secondVar = bestIterationTriple.getSecond();
            if(firstVar.isLatentVariable())
                latentVariables.add(firstVar.getName());
            if(secondVar.isLatentVariable())
                latentVariables.add(secondVar.getName());

            /* 1.3 - Estimate their cardinality and modify currentSet */
            if(bestIterationTriple.getThird().getName().equals("AddDiscreteNode")){
                // Obtenemos el padre de ambas variables y lo añadimos
                DiscreteVariable newLatentVar = (DiscreteVariable) bestIterationResult.getBayesianNetwork()
                        .getNode(firstVar)
                        .getParents().stream()
                        .findFirst().get().getContent();
                latentVariables.add(newLatentVar.getName());

                bestIterationResult = estimateLocalCardinality(latentVariables, bestIterationResult, data);
                //newLatentVar = bestIterationResult.getBayesianNetwork().getNode(newLatentVar.getName()).getVariable();

                currentSet.remove(firstVar.getName());
                currentSet.remove(secondVar.getName());
                currentSet.add(newLatentVar.getName());

            } else if(bestIterationTriple.getThird().getName().equals("AddArc")){
                bestIterationResult = estimateLocalCardinality(latentVariables, bestIterationResult, data);
                //secondVar = bestIterationResult.getBayesianNetwork().getNode(secondVar.getName()).getVariable();
                currentSet.remove(secondVar.getName());
            }

            /* 1.4 - Then, if allowed, we globally learn the parameters of the resulting model */
            if(this.iterationGlobalEM) {
                EM iterationEM = new EM(this.iterationEMConfig);
                bestIterationResult = iterationEM.learnModel(bestIterationResult.getBayesianNetwork(), data);
            }

            LogUtils.info("\nIteration["+iteration+"] = "+bestIterationTriple.getThird().getName() +
                    "(" + bestIterationTriple.getFirst().getName() + ", " + bestIterationTriple.getSecond().getName() + ") -> " + bestIterationResult.getScoreValue(), logLevel);

            /* En caso de que la iteracion no consiga mejorar el score del modelo, paramos el bucle */
            if(bestIterationResult.getScoreValue() <= bestResult.getScoreValue()) {
                LogUtils.debug("Doesn't improve the score: " + bestIterationResult.getScoreValue() + " <= " + bestResult.getScoreValue() + " (old best)", logLevel);
                LogUtils.debug("--------------------------------------------------", logLevel);
                keepsImproving = false;
            } else {
                LogUtils.debug("Improves the score: " + bestIterationResult.getScoreValue() + " > " + bestResult.getScoreValue() + " (old best)", logLevel);
                LogUtils.debug("--------------------------------------------------", logLevel);
                bestResult = bestIterationResult;
            }
        }

        /* 2 - We gblobally learn the parameters of the resulting model */
        EM finalEM = new EM(this.finalEMConfig);
        bestResult = finalEM.learnModel(bestResult.getBayesianNetwork(), data);
        LogUtils.info("\nFinal score after global EM: " + bestResult.getScoreValue(), logLevel);
        return bestResult;
    }

    private LearningResult<DiscreteBayesNet> estimateLocalCardinality(List<String> latentVariables,
                                                                      LearningResult<DiscreteBayesNet> currentBestResult,
                                                                      DiscreteData data) {

        LfmIncIncreaseCard increaseCardOperator = new LfmIncIncreaseCard(this.localEMConfig, this.typeLocalEM);
        LfmIncDecreaseCard decreaseCardOperator = new LfmIncDecreaseCard(this.localEMConfig, this.typeLocalEM);
        LearningResult<DiscreteBayesNet> bestResult = new LearningResult<>(
                currentBestResult.getBayesianNetwork(),
                currentBestResult.getScoreValue(),
                currentBestResult.getScoreType(),
                currentBestResult.getName());

        while(true) {

            LearningResult<DiscreteBayesNet> increaseCardResult = increaseCardOperator.apply(latentVariables, currentBestResult.getBayesianNetwork(), data);
            LearningResult<DiscreteBayesNet> decreaseCardResult = decreaseCardOperator.apply(latentVariables, currentBestResult.getBayesianNetwork(), data);

            if(increaseCardResult.getScoreValue() > decreaseCardResult.getScoreValue() && increaseCardResult.getScoreValue() > bestResult.getScoreValue())
                bestResult = increaseCardResult;
            else if(decreaseCardResult.getScoreValue() > increaseCardResult.getScoreValue() && decreaseCardResult.getScoreValue() > bestResult.getScoreValue())
                bestResult = decreaseCardResult;
            else
                return bestResult;
        }
    }
}
