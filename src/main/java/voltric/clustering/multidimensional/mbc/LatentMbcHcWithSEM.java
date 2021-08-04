package voltric.clustering.multidimensional.mbc;

import voltric.clustering.multidimensional.mbc.operator.*;
import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.structure.latent.StructuralEM;
import voltric.model.DiscreteBayesNet;
import voltric.variables.DiscreteVariable;

import java.util.Set;

/**
 * Algoritmo hill-climbing diseñado para trabajar con modelos latentes. Si bien sus operadores actualmente han sido pensados
 * para trbabajar con modelos MBC, lo cierto es que vale para cualquier tipo de estructura, solo las restricciones que
 * impongamos a su estructura y el comportamiento de los operadores "AddLatentNode" y "RemoveLatentNode" hara que
 * se restrinja a ese tipo de modelos.
 *
 * Este algoritmo se basa en la combinacion de 4 operadores para trabajar con variables latentes:
 * - IncreaseLatentCardinality
 * - DecreaseLatentCardinality
 * - AddLatentNode
 * - RemoveLatentNode
 *
 * Las operaciones de arcos se realizan puramente con el SEM. De esta forma, cuando se añade un nuevo nodo latente o se
 * incrementa la cardinalidad de uno de ellos, se llama al proceso SEM en vez de al EM, tomando dicho modelo modificado
 * como estructura inicial.
 *
 * NOTA: En los metodos se pide que se pase una instancia del Structural EM ya que dicho objeto contiene las restricciones
 * que se van a imponer con respecto a los arcos del modelo (por ejemplo si quiero aprender un OLCM, un HLCM o algo diferente)
 *
 * NOTA 2: Podria modificarse para que se dividiese en expansion y simplificacion
 *
 * NOTA 3: Si el algoritmo SEM deja una LV con un solo hijo se podria eliminar automaticamente por lo que dice Pearl
 * o ver si el propio algoritmo lo hace (que deberia, ya que son parametros absurdos).
 */
public class LatentMbcHcWithSEM {

    private Set<LatentMbcHcOperator> operators;

    private int maxIterations;

    private double threshold;

    public LatentMbcHcWithSEM(int maxIterations,
                              double threshold,
                              Set<LatentMbcHcOperator> operators) {

        this.operators = operators;
        this.maxIterations = maxIterations;
        this.threshold = threshold;
    }

    public LearningResult<DiscreteBayesNet> learnModel(DiscreteBayesNet seedNet, DiscreteData data, StructuralEM sem) {

        /* Clonamos el modelo base para no modificarlo con el EM */
        DiscreteBayesNet clonedNet = seedNet.clone();

        /* Comenzamos realizando una iteracion inicial del SEM para obtener el score base */
        LearningResult<DiscreteBayesNet> previousIterationResult = sem.getEm().learnModel(clonedNet, data); // EM inicial para el SEM (deberia ser su primer paso)
        previousIterationResult = sem.learnModel(previousIterationResult, data);

        int iterations = 0;

        /* Bucle general */
        while (iterations < this.maxIterations) {
            iterations = iterations + 1;

            System.out.println("iteration: " + iterations);

            LearningResult<DiscreteBayesNet> iterationBestResult = previousIterationResult;
            LatentMbcHcOperator iterationBestOperator = null;

            // 1 - Iteramos por el conjunto de operadores y nos quedamos con el que poseea mejor score
            for(LatentMbcHcOperator operator : this.operators){
                LearningResult<DiscreteBayesNet> result = operator.apply(previousIterationResult.getBayesianNetwork(), data, sem);
                if(result.getScoreValue() > iterationBestResult.getScoreValue()) {
                    iterationBestResult = result;
                    iterationBestOperator = operator;
                }
            }

            // 2 - Devuelve el modelo previo si la iteracion actual no ha conseguido mejorar el score lo suficiente
            if(previousIterationResult.getScoreValue() >= iterationBestResult.getScoreValue() ||
                    Math.abs(iterationBestResult.getScoreValue() - previousIterationResult.getScoreValue()) < threshold) {
                return new LearningResult<>(
                        previousIterationResult.getBayesianNetwork(),
                        previousIterationResult.getScoreValue(),
                        previousIterationResult.getScoreType());
            }

            System.out.println("best Operator: " + iterationBestOperator.getClass().getSimpleName());

            // 3 - Si se ha añadido o eliminado una variable latente, actualizamos las restricciones del modelo MBC
            updateSEM(sem, iterationBestOperator, previousIterationResult.getBayesianNetwork(), iterationBestResult.getBayesianNetwork());

            // 4 - Si el modelo de la iteracion mejora lo suficiente el score, se almacena
            previousIterationResult = iterationBestResult;

        }

        /*
            En caso de que el numero de iteraciones se haya superado mientras el modelo mejoraba de forma satisfactoria,
            se devuelve el resultado de la ultima iteracion realizada.
         */
        return previousIterationResult;
    }

    private void updateSEM(StructuralEM sem, LatentMbcHcOperator iterationBestOperator, DiscreteBayesNet seedNet, DiscreteBayesNet iterationBestNet) {

        /* Para el caso de que se haya añadido un nuevo nodo latente, actualizamos el SEM con dicha nueva variable */
        if(iterationBestOperator instanceof AddLatentNode){

            // 1 - Buscamos la nueva variable latente
            DiscreteVariable newLatentVariable = iterationBestNet.getLatentVariables().stream()
                    .filter(x->!seedNet.getLatentVariables().contains(x))
                    .findFirst().get();

            // 2 - Añadimos restricciones sobre esa variable en el SEM
            sem.addLatentVar(newLatentVariable);

        /* Para el caso en que se haya eliminado un nodo latente, actualizamos el SEM con dicha variable */
        } else if(iterationBestOperator instanceof RemoveLatentNode){

            // 1 - Buscamos la variable latente eliminada
            DiscreteVariable removedLatentVariable = seedNet.getLatentVariables().stream()
                    .filter(x->!iterationBestNet.getLatentVariables().contains(x))
                    .findFirst().get();

            // 2 - Eliminamos las restricciones de dicha variable en el SEM
            sem.removeLatentVar(removedLatentVariable);
        } else if(iterationBestOperator instanceof IncreaseLatentCardinality || iterationBestOperator instanceof DecreaseLatentCardinality) {

            // 1 - Buscamos la variable latente cuya cardinalidad ha cambiado
            DiscreteVariable newLatentVariable = iterationBestNet.getLatentVariables().stream()
                    .filter(x->!seedNet.getLatentVariables().contains(x))
                    .findFirst().get();

            // 2 - Buscamos en la seedNet su equivalente con la cardinalidad antigua
            DiscreteVariable oldLatentVariable = seedNet.getLatentVariables().stream()
                    .filter(x->x.getName().equals(newLatentVariable.getName()))
                    .findFirst().get();

            // 3 - Actualizamos las restricciones sobre ambas variables en el SEM
            sem.addLatentVar(newLatentVariable);
            sem.removeLatentVar(oldLatentVariable);
        }
    }
}
