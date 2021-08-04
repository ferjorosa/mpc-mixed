package voltric.clustering.multidimensional;


import voltric.clustering.unidimensional.LearnLatentLCM;
import voltric.data.DiscreteData;
import voltric.graph.Edge;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.learning.structure.hillclimbing.global.DecreaseLatentCardinality;
import voltric.learning.structure.hillclimbing.global.GlobalHillClimbing;
import voltric.learning.structure.hillclimbing.global.HcOperator;
import voltric.learning.structure.hillclimbing.global.IncreaseLatentCardinality;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.model.HLCM;
import voltric.model.creator.HlcmCreator;
import voltric.util.distance.Hellinger;
import voltric.util.information.mi.NMI;
import voltric.util.information.mi.normalization.NMImax;
import voltric.variables.DiscreteVariable;
import voltric.variables.Variable;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Created by equipo on 18/04/2018.
 *
 * Distingo 2 metodos privados:
 *
 *      - El de añadir cada variable al LCM ya existente para ver si mejora su Hellinger
 *      - El de formar un LCM entre las 2 variables con mayor NMI del vector actual
 *
 * TODO: Habria que probar que funciona bien el metodo y si es asi, eliminar comentarios innecesarios
 */
public class StaticHellingerFinder {


    private static double thresholdParaEM = 0.5;

    private static int maxCardinality = 10;

    private static double nmiThreshold = 0.05; // si la NMI de dos variables no supera este threshold, se consideran independientes

    private static double hellingerThreshold = 0.02;

    public static DiscreteBayesNet find(DiscreteData data, AbstractEM em) {

        /** Antes de comenzar, calculamos la NMI entre cada par de variables del dataSet */
        Map<DiscreteVariable, Map<DiscreteVariable, Double>> pairValues = NMI.computePairwise(data.getVariables(), data, new NMImax());

        List<DiscreteVariable> currentVariables = new ArrayList<>();
        currentVariables.addAll(data.getVariables());

        Map<List<DiscreteVariable>, Double> currentPartitionScores = new HashMap<>();
        Map<List<DiscreteVariable>, HLCM> currentPartitions = new HashMap<>();

        /** Iteramos hasta que el numero de variables del vector se encuentre vacio */
        while(currentVariables.size() > 0) {

            /** Si no existe ningun LCM, creamos un nuevo con el par de variables que tienen mejor valor NMI */
            if(currentPartitions.size() == 0){

                if(!lcmCanBeFormed(currentVariables, pairValues)) {
                    currentVariables = new ArrayList<>(); // We empty the vector of currentVariables

                // En el caso de que si exista un par de variables en el vector cuya NMI supere el threshold
                } else {
                    HLCM best_lcm = formBestLCM(data, em, currentVariables, pairValues, currentPartitions.size());
                    double averageHellingerClusterDist = Hellinger.averageClusterDistancesLCM(best_lcm);
                    currentPartitionScores.put(best_lcm.getManifestVariables(), averageHellingerClusterDist);
                    currentPartitions.put(best_lcm.getManifestVariables(), best_lcm);
                    currentVariables.removeAll(best_lcm.getManifestVariables());
                }
            }
            /** En el caso de que existan uno o mas, probamos a añadir cada una de las variables a cada uno de los LCM
             * y nos quedamos con aquel nuevo modelo que mejore más la distancia de Hellinger entre los clusters de los LCMs
             */
            else {

                DiscreteVariable varThatIncreasesHellingerClusterDist = improveLcmHellinger(data, em, currentVariables, currentPartitions, currentPartitionScores);
                if (varThatIncreasesHellingerClusterDist != null)
                    currentVariables.remove(varThatIncreasesHellingerClusterDist);
                /** En el caso de que no exista ninguna variable que mejore la calidad de los LCM existentes, aprendemos un "bestLCM",
                 * formando una nueva particion
                 */
                else {
                    // siempre y cuando las variables restantes no sean independientes
                    if(!lcmCanBeFormed(currentVariables, pairValues)) {
                        currentVariables = new ArrayList<>();
                    } else {
                        HLCM best_lcm = formBestLCM(data, em, currentVariables, pairValues, currentPartitions.size());
                        double averageHellingerClusterDist = Hellinger.averageClusterDistancesLCM(best_lcm);
                        currentPartitionScores.put(best_lcm.getManifestVariables(), averageHellingerClusterDist);
                        currentPartitions.put(best_lcm.getManifestVariables(), best_lcm);
                        currentVariables.removeAll(best_lcm.getManifestVariables());
                    }

                }
            }
        }

        /** Finalmente combinamos los LCMs con las variables independientes y formamos un modelo multidimensional
         *
         * Si existen variables que no pertenecen a ninguno de los LCMs, es que han sido seleccionadas como variables
         * de "ruido". Para ese caso, se añaden al modelo pero sin conexion directa con ninguna particion.
         */
        return createInitialOLCM(data, em, currentPartitions.values());
    }

    // TODO: El currentVariables se le pasa como argumento porque vendria dado por el metodo superior
    // TODO: IMPORTANTE-->  En vez de pasarle una lista con los currentLCMs, deberiamos pasarle un Map que asocie cada LCM con su score Hellinger medio
    /** Devuelve una variable != null, si ella ha conseguido aumentar la distancia de Hellinger de alguna particion */
    private static DiscreteVariable improveLcmHellinger(DiscreteData data,
                                                        AbstractEM em,
                                                        List<DiscreteVariable> currentVariables,
                                                        Map<List<DiscreteVariable>, HLCM> currentPartitions,
                                                        Map<List<DiscreteVariable>, Double> currentPartitionScores) {


        /** Copiamos los LCMs para no modificarlos */
        List<HLCM> currentLCMsCopied = new ArrayList<>();
        for (HLCM lcm : currentPartitions.values())
            currentLCMsCopied.add(lcm.clone());

        double bestHellingerIncrease = 0;
        double bestHellingerScore = 0;
        DiscreteVariable bestVariable = null;
        HLCM bestLCM = null;
        List<DiscreteVariable> bestOldSetOfVariables = new ArrayList<>(); // The Map key

        /** Iteramos por el conjunto de currentVars y por el conjunto de currentPartitions */
        for (DiscreteVariable variable : currentVariables){
            for (HLCM lcmCopy : currentLCMsCopied) {
                List<DiscreteVariable> oldManifestVariables = lcmCopy.getManifestVariables();
                DiscreteBeliefNode varNode = lcmCopy.addNode(variable);
                Edge<Variable> newVarEdge = lcmCopy.addEdge(varNode, lcmCopy.getRoot());
                HLCM lcmCopyWithVar = increaseAndDecreaseCardinality(data, em, lcmCopy);
                double hellingerOfLcmCopyWithVar = Hellinger.averageClusterDistancesLCM(lcmCopyWithVar);

                double hellingerIncrease = hellingerOfLcmCopyWithVar - currentPartitionScores.get(oldManifestVariables);
                if (hellingerIncrease > bestHellingerIncrease) {
                    bestHellingerIncrease = hellingerIncrease;
                    bestHellingerScore = hellingerOfLcmCopyWithVar;
                    bestLCM = lcmCopyWithVar;
                    bestOldSetOfVariables = oldManifestVariables;
                    bestVariable = variable;
                }

                // Independientemente de si mejoro o no el score Hellinger, eliminamos el arco y el nodo añadidos a variable para el LCM en cuestion
                // Esto nos sirve para evitar que tengamos que clonar el LCM por cada variable
                lcmCopy.removeEdge(newVarEdge);
                lcmCopy.removeNode(varNode);
            }
        }

        /** En este caso, alguna de las variables ha mejorado la distancia de Hellinger entre clusters de alguna de las
         * particiones, por lo que la almacenamos como una nueva particion
         */
        if(bestHellingerIncrease > hellingerThreshold){
            currentPartitions.remove(bestOldSetOfVariables);
            currentPartitionScores.remove(bestOldSetOfVariables);

            currentPartitions.put(bestLCM.getManifestVariables(), bestLCM);
            currentPartitionScores.put(bestLCM.getManifestVariables(), bestHellingerScore);
            return bestVariable;
        }
        // El valor de bestVariable sera != null si el hellingerIncrease > 0; Independientemente se devuelve
        return null;
    }

    /** Devuelve true en caso de que exista al menos un pairValue > threshold, false otherwise */
    private static boolean lcmCanBeFormed(List<DiscreteVariable> currentVariables, Map<DiscreteVariable, Map<DiscreteVariable, Double>> pairValues) {

        /** Seleccionamos el par de variables tienen mayor NMI */
        DiscreteVariable bestFirstVar = null;
        DiscreteVariable bestSecondVar = null;
        double bestNmiValue = -1;

        /** The keyset is filtered according to the vector of current variables */
        for(DiscreteVariable firstVar: pairValues.keySet().stream().filter(x -> currentVariables.contains(x)).collect(Collectors.toList())){

            Map<DiscreteVariable, Double> secondVarsWithValues = pairValues.get(firstVar);

            for(DiscreteVariable secondVar: secondVarsWithValues.keySet().stream().filter(x -> currentVariables.contains(x)).collect(Collectors.toList())){
                double value = secondVarsWithValues.get(secondVar);
                if(value > bestNmiValue){
                    bestFirstVar = firstVar;
                    bestSecondVar = secondVar;
                    bestNmiValue = value;
                }
            }
        }

        /** Eliminamos las variables escogidas del vector de variables */
        List<DiscreteVariable> bestVariables = new ArrayList<>(2);
        bestVariables.add(bestFirstVar);
        bestVariables.add(bestSecondVar);

        return bestNmiValue > nmiThreshold;
    }

    private static HLCM formBestLCM(DiscreteData data, AbstractEM em, List<DiscreteVariable> currentVariables, Map<DiscreteVariable, Map<DiscreteVariable, Double>> pairValues, int lcmIndex) {

        /** Despues seleccionamos el par de variables con mayor NMI para que formen una particion entre ambos */
        DiscreteVariable bestFirstVar = null;
        DiscreteVariable bestSecondVar = null;
        double bestNmiValue = -1;

        /** The keyset is filtered according to the vector of current variables */
        for(DiscreteVariable firstVar: pairValues.keySet().stream().filter(x -> currentVariables.contains(x)).collect(Collectors.toList())){

            Map<DiscreteVariable, Double> secondVarsWithValues = pairValues.get(firstVar);

            for(DiscreteVariable secondVar: secondVarsWithValues.keySet().stream().filter(x -> currentVariables.contains(x)).collect(Collectors.toList())){
                double value = secondVarsWithValues.get(secondVar);
                if(value > bestNmiValue){
                    bestFirstVar = firstVar;
                    bestSecondVar = secondVar;
                    bestNmiValue = value;
                }
            }
        }

        /** Eliminamos las variables escogidas del vector de variables */
        List<DiscreteVariable> bestVariables = new ArrayList<>(2);
        bestVariables.add(bestFirstVar);
        bestVariables.add(bestSecondVar);

        /** Formamos un LCM con las 2 mejores variables cuya cardinalidad se estima de forma greedy */
        String latentVarName = "clustVar_" + lcmIndex;
        HLCM lcm2Vars = HlcmCreator.createLCM(bestVariables, 2, "LCM", latentVarName, new Random());
        lcm2Vars = (HLCM) LearnLatentLCM.learnModelToMaxCardinality(lcm2Vars, data.project(bestVariables), em, thresholdParaEM, maxCardinality).getBayesianNetwork();

        return lcm2Vars;
    }

    /** Este metodo genera 2 modelos:
     *      1 - Un modelo a partir de un proceso de HillClibing con solo incremento de cardinalidad
     *      2 - Un modelo a partir de un proceso de HillClimbing con solo decremento de cardinalidad
     *
     * Lo hago separado para que no repita modelos pero evalue las 2 posibilidades
     * @return
     */
    private static HLCM increaseAndDecreaseCardinality(DiscreteData data, AbstractEM em, HLCM lcm) {

        /** IncreaseCardinality-only hill-climbing process */
        IncreaseLatentCardinality ilcOperator = new IncreaseLatentCardinality(maxCardinality);
        Set<HcOperator> ilcOperatorSet = new HashSet<>();
        ilcOperatorSet.add(ilcOperator);

        GlobalHillClimbing hillClimbingIncrease = new GlobalHillClimbing(ilcOperatorSet, 500, thresholdParaEM);

        /** DecreaseCardinality-only hill-climbing process */
        DecreaseLatentCardinality dlcOperator = new DecreaseLatentCardinality();
        Set<HcOperator> dlcOperatorSet = new HashSet<>();
        dlcOperatorSet.add(dlcOperator);

        GlobalHillClimbing hillClimbingDecrease = new GlobalHillClimbing(dlcOperatorSet, 500, thresholdParaEM);

        LearningResult<DiscreteBayesNet> bestIncreaseLcmResult = hillClimbingIncrease.learnModel(lcm, data, em);
        LearningResult<DiscreteBayesNet> bestDecreaseLcmResult = hillClimbingDecrease.learnModel(lcm, data, em);

        if(bestIncreaseLcmResult.getScoreValue() > bestDecreaseLcmResult.getScoreValue())
            return (HLCM) bestIncreaseLcmResult.getBayesianNetwork();
        else
            return (HLCM) bestDecreaseLcmResult.getBayesianNetwork();
    }

    /** Para quitarnos de problemas lo vamos a generar copiando la estructura y reaprendiendo el modelo con EM, sin copiar CPTs o similar */
    private static DiscreteBayesNet createInitialOLCM(DiscreteData data, AbstractEM em, Collection<HLCM> partitions) {

        DiscreteBayesNet initialOLCM = new DiscreteBayesNet("olcm - " + data.getName());

        /** Primero añadimos todas las variables de los datos como MVs */
        for(DiscreteVariable variable: data.getVariables())
            initialOLCM.addNode(variable);

        /** Iteramos por lo HLCMs y añadimos tanto sus LVs como los arcos de ellas a sus MVs asociadas */
        for(HLCM partition: partitions){
            DiscreteBeliefNode newPartitionNode = initialOLCM.addNode(partition.getRoot().getVariable());
            for(DiscreteVariable manifestVar: partition.getManifestVariables())
                initialOLCM.addEdge(initialOLCM.getNode(manifestVar), newPartitionNode);
        }

        /** Aprendemos el modelo multidimensional completo*/
        LearningResult<DiscreteBayesNet> result = em.learnModel(initialOLCM, data);
        System.out.println(result.getScoreValue());

        return result.getBayesianNetwork();
    }
}
