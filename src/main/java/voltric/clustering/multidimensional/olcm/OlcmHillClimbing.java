package voltric.clustering.multidimensional.olcm;

import voltric.clustering.multidimensional.olcm.operator.OlcmHcOperator;
import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.model.DiscreteBayesNet;

import java.util.Set;

/**
 * Algoritmo hill-climbing diseñado para trabajar con modelos OLCM. Fue desarrollado para el PGM 2018.
 *
 * Inspirado por el trabajo de Chen et al. (2012), se compone de 2 fases:
 * - Expansión de la estructura.
 * - Simplificación de la estructura.
 *
 * Tal y como se encuentra desarrollado el operador de añadir nodo latente, no se puede partir mas que de una estructura OLCM.
 * En el articulo del PGM, el metodo de inicializacion se basa en la seleccion de grupos de atributos mediante informacion
 * mutua y distancia de Hellinger.
 *
 * Hay un problema que no se comenta en dicho paper y es que la manera en la que fue implementado AddLatentNode complica
 * el hecho de iniciar con una estructura LCM, para ese caso tiene quizas mas sentido un operador similar al de Chen.
 *
 * La razon es que en la primera version el operador {@link voltric.clustering.multidimensional.olcm.operator.OldAddOlcmNode}
 * permitia crear un nodo LV entre MVs de la misma particion, pero eso añadia mucho tiempo de ejecucion y muchas veces
 * dichas LVs carecian de sentido, por ello, creamos un nuevo operador que SOLO permitia crear un nodo LV entre nodos de
 * diferentes particiones, lo que restringe a trabajar con  modelos multidimensionales (multiples particiones, cada una
 * de ellas con una variable latente).
 *
 */
public class OlcmHillClimbing {

    private Set<OlcmHcOperator> expansionOperators;

    private Set<OlcmHcOperator> simplificationOperators;

    private int maxIterations;

    private double threshold;

    public OlcmHillClimbing(int maxIterations,
                            double threshold,
                            Set<OlcmHcOperator> expansionOperators,
                            Set<OlcmHcOperator> simplificationOperators){
        /** Expansion Operators */
        this.expansionOperators =expansionOperators;

        /** Simplification Operators */
        this.simplificationOperators =simplificationOperators;

        this.maxIterations = maxIterations;
        this.threshold = threshold;
    }

    public LearningResult<DiscreteBayesNet> learnModel(DiscreteBayesNet seedNet, DiscreteData data, AbstractEM em) {

        LearningResult<DiscreteBayesNet> previousIterationResult = em.learnModel(seedNet.clone(), data);

        int iterations = 0;

        // Bucle general que itera por las fases de expansin y simplificacion
        while (iterations < this.maxIterations) {
            System.out.println("iterations (" + iterations + ") score: " + previousIterationResult.getScoreValue());
            iterations = iterations + 1;

            /** Expansion process */
            LearningResult<DiscreteBayesNet> expansionBestResult = expansionProcess(previousIterationResult, data, em);

            // If the expansion process haven't increased the score, we stop the HC algorithm
            if (previousIterationResult.getScoreValue() >= expansionBestResult.getScoreValue()
                    || Math.abs(expansionBestResult.getScoreValue() - previousIterationResult.getScoreValue()) < threshold)
                return previousIterationResult;

            /** Simplification process */
            LearningResult<DiscreteBayesNet> simplificationBestResult = simplificationProcess(expansionBestResult, data, em);

            // If the simplification process haven't increased the score, we stop the HC algorithm
            if (expansionBestResult.getScoreValue() >= simplificationBestResult.getScoreValue()
                    || Math.abs(expansionBestResult.getScoreValue() - simplificationBestResult.getScoreValue()) < threshold)
                return expansionBestResult;

            // If both the expansion and simplification processes have improved the model's score, the model is stored in memory
            previousIterationResult = simplificationBestResult;
        }

        // Si se ha pasado el maximo de iteraciones y segui mejorando el score, lo corto el HC y devuelvo el modelo
        return previousIterationResult;
    }

    private LearningResult<DiscreteBayesNet> expansionProcess (final LearningResult<DiscreteBayesNet> previousIterationResult, DiscreteData data, AbstractEM em) {

        LearningResult<DiscreteBayesNet> expansionBestResult = previousIterationResult;
        LearningResult<DiscreteBayesNet> previousResult;

        do {
            System.out.println("Expansion score: " + expansionBestResult.getScoreValue());

            previousResult = expansionBestResult;
            for (OlcmHcOperator operator : this.expansionOperators) {
                LearningResult<DiscreteBayesNet> expansionResult = operator.apply(expansionBestResult.getBayesianNetwork(), data, em);
                if (expansionResult.getScoreValue() > expansionBestResult.getScoreValue()) {
                    expansionBestResult = expansionResult;
                }
            }
        } while (expansionBestResult.getScoreValue() > previousResult.getScoreValue());

        return expansionBestResult;
    }

    private LearningResult<DiscreteBayesNet> simplificationProcess (final LearningResult<DiscreteBayesNet> expansionBestResult, DiscreteData data, AbstractEM em) {

        LearningResult<DiscreteBayesNet> simplificationBestResult = expansionBestResult;
        LearningResult<DiscreteBayesNet> previousResult;

        do {
            System.out.println("Simplification score: " + simplificationBestResult.getScoreValue());

            previousResult = simplificationBestResult;
            for (OlcmHcOperator operator : this.simplificationOperators) {
                LearningResult<DiscreteBayesNet> simplificationResult = operator.apply(simplificationBestResult.getBayesianNetwork(), data, em);
                if (simplificationResult.getScoreValue() > simplificationBestResult.getScoreValue()) {
                    simplificationBestResult = simplificationResult;
                }
            }
        } while (simplificationBestResult.getScoreValue() > previousResult.getScoreValue());

        return simplificationBestResult;
    }
}
