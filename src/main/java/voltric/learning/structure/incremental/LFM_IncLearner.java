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
import voltric.learning.structure.incremental.util.IncLearnerUtils;
import voltric.model.DiscreteBayesNet;
import voltric.variables.DiscreteVariable;

import java.util.*;

/*
    A diferencia de AMIDST, en Voltric las variables se encuentran definidas por el nombre y la cardinalidad. Esto
    supone un problema ya que las modificaciones de cardinalidad en Voltric crean nuevos objetos y con ello las referencias
    en las colecciones dejan de encajar. Por eso, vamos a utilizar unicamente el nombre de la variable como identificador,
    esto es peligroso porque puede haber colisiones Hash, pero dado que vamos a usar pocas variables, no deberia suponer
    un gran problema.
 */
public class LFM_IncLearner {

    private Set<LfmIncOperator> operators;

    private boolean iterationGlobalEM;

    private boolean normalizedMI;

    private EmConfig initialEMConfig;

    private EmConfig localEMConfig;

    private EmConfig iterationEMConfig;

    private EmConfig finalEMConfig;

    private TypeLocalEM typeLocalEM;

    public LFM_IncLearner(Set<LfmIncOperator> operators,
                          boolean iterationGlobalEM,
                          boolean normalizedMI,
                          EmConfig initialEMConfig,
                          EmConfig localEMConfig,
                          EmConfig iterationEMConfig,
                          EmConfig finalEMConfig,
                          TypeLocalEM typeLocalEM) {

        this.operators = operators;
        this.iterationGlobalEM = iterationGlobalEM;
        this.normalizedMI = normalizedMI;
        this.initialEMConfig = initialEMConfig;
        this.localEMConfig = localEMConfig;
        this.iterationEMConfig = iterationEMConfig;
        this.finalEMConfig = finalEMConfig;
        this.typeLocalEM = typeLocalEM;
    }

    public LearningResult<DiscreteBayesNet> learnModel(DiscreteData data,
                                                       int alpha,
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

        /* 1 - Estimamos la MI entre cada par de atributos de los datos */
        Map<String, Map<String, Double>> currentMIsMatrix = new LinkedHashMap<>(data.getVariables().size());
        for(DiscreteVariable x: data.getVariables()) {
            currentMIsMatrix.put(x.getName(), new LinkedHashMap<>());
            for (DiscreteVariable y : data.getVariables())
                if (!x.equals(y))
                    currentMIsMatrix.get(x.getName()).put(y.getName(),IncLearnerUtils.mi(x,y, data));
        }

        /* Inicializamos la estructura auxiliar para la estimacion de la MI de variables latentes */
        Map<String, int[]> currentDataForMI = new HashMap<>(data.getVariables().size()); // Associated data to current set of variables. It will be used to estimate MI
        initializeDataForMI(currentDataForMI, data, currentSet);

        LogUtils.info("Initial score: " + bestResult.getScoreValue(), logLevel);

        /* 2 - Bucle principal */
        boolean keepsImproving = true;
        int iteration = 0;
        while(keepsImproving && currentSet.size() > 1) {

            iteration++;

            /* Estimamos los alpha pares de variables cuyo valor de MI es mas alto */
            PriorityQueue<Tuple3<String, String, Double>> selectedTriples = highestMiVariables(currentMIsMatrix, alpha);

            LearningResult<DiscreteBayesNet> bestIterationResult =
                    new LearningResult<>(null, -Double.MAX_VALUE, this.localEMConfig.getScoreType());
            Tuple3<DiscreteVariable, DiscreteVariable, LearningResult<DiscreteBayesNet>> bestIterationTriple =
                    new Tuple3<>(null, null, bestIterationResult);

            /* 1.1 - Iterate through the operators and select the one that returns the best model */
            for(LfmIncOperator operator: operators) {
                Tuple3<DiscreteVariable, DiscreteVariable, LearningResult<DiscreteBayesNet>> operatorTriple =
                        operator.apply(selectedTriples, bestResult.getBayesianNetwork(), data);

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

            String miVarName = "";

            /* 1.3 - Estimate their cardinality and modify currentSet */
            if(bestIterationTriple.getThird().getName().equals("AddDiscreteNode")){
                // Obtenemos el padre de ambas variables y lo añadimos
                DiscreteVariable newLatentVar = (DiscreteVariable) bestIterationResult.getBayesianNetwork()
                        .getNode(firstVar)
                        .getParents().stream()
                        .findFirst().get().getContent();
                latentVariables.add(newLatentVar.getName());

                bestIterationResult = estimateLocalCardinality(latentVariables, bestIterationResult, data);

                /* Preparamos la variable padre para su nueva estimacion de MI */
                miVarName = newLatentVar.getName();

                /* Actualizamos las estructuras de datos asociadas */
                removeVarFromCurrentDataStructures(firstVar.getName(), currentSet, currentDataForMI, currentMIsMatrix);
                removeVarFromCurrentDataStructures(secondVar.getName(), currentSet, currentDataForMI, currentMIsMatrix);
                currentSet.add(newLatentVar.getName());

            } else if(bestIterationTriple.getThird().getName().equals("AddArc")){
                bestIterationResult = estimateLocalCardinality(latentVariables, bestIterationResult, data);
                removeVarFromCurrentDataStructures(secondVar.getName(), currentSet, currentDataForMI, currentMIsMatrix);
                if(firstVar.isLatentVariable())
                    miVarName = firstVar.getName();
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
                if(!miVarName.equals(""))
                    estimateVariableMIs(currentMIsMatrix, data, currentDataForMI, bestResult.getBayesianNetwork(), miVarName);
            }
        }

        /* 2 - We gblobally learn the parameters of the resulting model */
        EM finalEM = new EM(this.finalEMConfig);
        bestResult = finalEM.learnModel(bestResult.getBayesianNetwork(), data);
        LogUtils.info("\nFinal score after global EM: " + bestResult.getScoreValue(), logLevel);
        return bestResult;
    }

    private void initializeDataForMI(Map<String, int[]> currentData, DiscreteData data, List<String> varNames) {
        /* Map initialization */
        for(String varName: varNames)
            currentData.put(varName, new int[data.getInstances().size()]);

        /* Introduce values into the Map */
        for(int instIndex = 0; instIndex < data.getInstances().size(); instIndex++) {
            int[] instData = data.getInstances().get(instIndex).getNumericValues();
            for(int varIndex = 0; varIndex < varNames.size(); varIndex++) {
                String varName = varNames.get(varIndex);
                currentData.get(varName)[instIndex] = instData[varIndex]; // Assign instanceValues to each variable
            }
        }
    }

    private class InverseMiComparator implements Comparator<Tuple3<String, String, Double>> {
        @Override
        public int compare(Tuple3<String, String, Double> o1, Tuple3<String, String, Double> o2) {
            if(o1.getThird().equals(o2.getThird()))
                return 0;
            // keep the biggest values
            return o1.getThird() > o2.getThird() ? 1 : -1;
        }
    }

    private PriorityQueue<Tuple3<String, String, Double>> highestMiVariables(Map<String, Map<String, Double>> misMatrix, int alpha) {

        PriorityQueue<Tuple3<String, String, Double>> queue = new PriorityQueue<>(alpha, new InverseMiComparator());

        /*
         * Creamos una lista con las keys del map para poder iterar por la "triangular" de la matriz.
         * Ademas, añadimos el primer elemento de la matriz a la queue para evitar tener que comprobar si la queue se encuentra vacia
         */
        List<String> keysList = new ArrayList<>(misMatrix.keySet());
        String firstKey = keysList.get(0);
        String secondKey = keysList.get(1);
        queue.add(new Tuple3<>(firstKey, secondKey, misMatrix.get(firstKey).get(secondKey)));

        /* Iteramos por la triangular de la matriz de MIs para obtener el par de variables con valor maximo */
        for(int i = 0; i < keysList.size(); i++)
            for(int j = i+1; j < keysList.size(); j++){
                String x = keysList.get(i);
                String y = keysList.get(j);

                if(queue.size() < alpha)
                    queue.add(new Tuple3<>(x,y,misMatrix.get(x).get(y)));
                else if(misMatrix.get(x).get(y) > queue.peek().getThird()){
                    queue.poll();
                    queue.add(new Tuple3<>(x,y,misMatrix.get(x).get(y)));
                }
            }

        return queue;
    }

    private void estimateVariableMIs(Map<String, Map<String, Double>> misMatrix,
                                     DiscreteData data,
                                     Map<String, int[]> dataForMI,
                                     DiscreteBayesNet model,
                                     String newVarName) {

        /* Completamos los datos de la variable y los almacenamos en dataForMI */
        int[] newVarData = IncLearnerUtils.predictData(data, model, newVarName);
        dataForMI.put(newVarName, newVarData);

        /* Estimamos la MI con las demas variables de misMatrix */
        misMatrix.put(newVarName, new LinkedHashMap<>());
        for(String varName: misMatrix.keySet()){
            int[][] bothVarsData = generate2dMatrix(dataForMI.get(varName), newVarData);
            double mi = IncLearnerUtils.mi(bothVarsData, this.normalizedMI);
            misMatrix.get(varName).put(newVarName, mi);
            misMatrix.get(newVarName).put(varName, mi);
        }
    }

    private int[][] generate2dMatrix(int[] firstVector, int[] secondVector) {
        int[][] matrix = new int[firstVector.length][2];

        for(int i = 0; i < firstVector.length; i++) {
            matrix[i][0] = firstVector[i];
            matrix[i][1] = secondVector[i];
        }

        return matrix;
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

    /** Elimina una variable de: currentSet, currentDataForMI y currentMIsMatrix */
    private void removeVarFromCurrentDataStructures(String varToRemoveName,
                                                    List<String> currentSet,
                                                    Map<String, int[]> currentDataForMI,
                                                    Map<String, Map<String, Double>> currentMIsMatrix) {
        currentSet.remove(varToRemoveName);
        currentDataForMI.remove(varToRemoveName);
        currentMIsMatrix.remove(varToRemoveName);
        for(String varName: currentMIsMatrix.keySet())
            currentMIsMatrix.get(varName).remove(varToRemoveName);
    }
}
