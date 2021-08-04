package voltric.learning.structure.hillclimbing.global;

import voltric.clustering.multidimensional.mbc.LatentMbcHcWithSEM;
import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.parameter.DiscreteParameterLearning;
import voltric.learning.structure.DiscreteStructureLearning;
import voltric.model.DiscreteBayesNet;

import java.util.Set;

/**
 * Este metodo "tradicional" de hill-climbing es el que utilizo como base en {@link LatentMbcHcWithSEM}
 * donde sustituyo el EM por SEM y añado dos operadores extra para:
 * - Añadir variable latente
 * - Eliminar variable latente
 *
 * TODO: TreeStructure no funciona correctamente ya que la llamada a isTree puede dar problemas al no trabajar con arboles incrementales
 */
public class GlobalHillClimbing implements DiscreteStructureLearning{

    private Set<HcOperator> operators;

    private int maxIterations;

    private double threshold;

    public GlobalHillClimbing(Set<HcOperator> operators, int maxIterations, double threshold){
        this.operators = operators;
        this.maxIterations = maxIterations;
        this.threshold = threshold;
    }


    /**
     *
     *
     * @param seedNet
     * @param data
     * @param parameterLearning
     * @return
     */
    //TODO: Nota: En principio no hago comprobaciones de que la black list
    public LearningResult<DiscreteBayesNet> learnModel(DiscreteBayesNet seedNet, DiscreteData data, DiscreteParameterLearning parameterLearning){

        LearningResult<DiscreteBayesNet> previousNetResult = parameterLearning.learnModel(seedNet.clone(), data);
        DiscreteBayesNet previousNet = previousNetResult.getBayesianNetwork();
        double previousScore = previousNetResult.getScoreValue();

        DiscreteBayesNet iterationBestNet = previousNet; // The BN whose applied operator returns the max score
        double iterationBestScore = previousScore;

        int iterations = 0;

        while(iterations < this.maxIterations){
            iterations = iterations + 1;

            //System.out.println("iterations: "+ iterations);

            // Iteration thourgh all the operators, where Associated BN with the best score is stored
            for(HcOperator operator : this.operators){
                LearningResult<DiscreteBayesNet> result = operator.apply(previousNet, data, parameterLearning);
                if(result.getScoreValue() > iterationBestScore) {
                    iterationBestNet = result.getBayesianNetwork();
                    iterationBestScore = result.getScoreValue();
                }
            }

            // Return previous model if current iteration didnt improve the score or the improvement wasn't enough to surpass the threshold
            if(previousScore >= iterationBestScore || Math.abs(iterationBestScore - previousScore) < threshold)
                return new LearningResult<>(previousNet, previousScore, parameterLearning.getScoreType());

            // Current iteration is now the previous one
            previousNet = iterationBestNet;
            previousScore = iterationBestScore;
        }

        // If the number of iterations have been surpassed and the model kept improving surpassing the threshold,
        // the last iteration's model is returned
        return new LearningResult<>(iterationBestNet, iterationBestScore, parameterLearning.getScoreType());
    }
}
