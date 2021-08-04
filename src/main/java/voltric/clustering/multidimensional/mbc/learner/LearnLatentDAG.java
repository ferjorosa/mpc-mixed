package voltric.clustering.multidimensional.mbc.learner;

import voltric.clustering.multidimensional.HellingerFinder;
import voltric.clustering.multidimensional.mbc.LatentMbcHcWithSEM;
import voltric.clustering.multidimensional.mbc.operator.*;
import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.learning.structure.latent.StructuralEM;
import voltric.learning.structure.type.DagStructure;
import voltric.model.DiscreteBayesNet;
import voltric.variables.DiscreteVariable;

import java.util.*;

public class LearnLatentDAG {

    /**
     * Version clasica con los 4 operadore donde se le puede pasar una estructura inicial especifica:
     * - LCM
     * - Multiples LCMs generados por un Finder
     * - Modelo multidimensional especifico
     * - etc.
     */
    public static LearningResult<DiscreteBayesNet> learnModel(DiscreteData data,
                                                              DiscreteBayesNet initialStructure,
                                                              double threshold,
                                                              AbstractEM em,
                                                              int initialLatentVarCardinality,
                                                              int maxLatentVarCardinality,
                                                              int minLatentVarCardinality) {

        List<DiscreteVariable> classVars = initialStructure.getLatentVariables();
        List<DiscreteVariable> featureVars = initialStructure.getManifestVariables();

        /* 1 - Crear la lista de arcos prohibidos a añadir (no se pueden añadir arcos entre MVs) */
        Map<DiscreteVariable, List<DiscreteVariable>> extraForbiddenAddArcs = new HashMap<>();
        for(DiscreteVariable featureVar: featureVars)
            extraForbiddenAddArcs.put(featureVar, new ArrayList<>(initialStructure.getManifestVariables())); // Lo pasamos como copia por si acaso, creo que no se modifica pero igual

        // Nota: Aquellos arcos que no se pueden añadir no pueden por tanto revertirse, por lo que no es necesario incluir codigo para ello.

        /* 2 - Crear el Structural EM */
        StructuralEM sem = new StructuralEM(classVars, featureVars,
                new HashMap<>(),
                extraForbiddenAddArcs, // Son los arcos que no se pueden añadir ADEMAS de los propios de una estructura MBC
                new HashMap<>(),
                new DagStructure(),
                em,
                Integer.MAX_VALUE, // Max iterations
                Integer.MAX_VALUE); // Max number of parents

        /* 3 - Crear los operadores del latent Hill-climber con SEM */
        Set<LatentMbcHcOperator> latentMbcHcOperators = new LinkedHashSet<>();
        latentMbcHcOperators.add(new AddLatentNode(initialLatentVarCardinality)); // Le pasamos la cardinalidad inicial de una nueva LV
        latentMbcHcOperators.add(new RemoveLatentNode());
        latentMbcHcOperators.add(new IncreaseLatentCardinality(maxLatentVarCardinality)); // Le pasamos la cardinalidad maxima de una LV
        latentMbcHcOperators.add(new DecreaseLatentCardinality(minLatentVarCardinality)); // Le pasamos la cardinalidad minima de una LV

        /* 4 - Crear el MBC Hill-climber con SEM */
        LatentMbcHcWithSEM latentMbcHillClimbingWithSEM = new LatentMbcHcWithSEM(Integer.MAX_VALUE, threshold, latentMbcHcOperators);

        /* 5 - Aprender el modelo latente MBC */
        return latentMbcHillClimbingWithSEM.learnModel(initialStructure, data, sem);
    }

    /**
     * Versión "MUY RAPIDA" donde se omiten los operadores de añadir o eliminar nodos. Se genera un modelo inicial mediante
     * el Hellinger Finder donde se establece el numero de variables latentes y luego se adapta los arcos del modelo
     * y la cardinalidad de los mismos.
     */
    public static LearningResult<DiscreteBayesNet> learnFastModelWithHellinger(DiscreteData data,
                                                                               HellingerFinder hellingerFinder,
                                                                               double threshold,
                                                                               AbstractEM em,
                                                                               int maxLatentVarCardinality,
                                                                               int minLatentVarCardinality) {

        /* 1 - Generamos la estructura inicial mediante el HellingerFinder */
        double initHellingerTime = System.currentTimeMillis();
        DiscreteBayesNet initialStructure = hellingerFinder.find(data, em);
        double endHellingerTime = System.currentTimeMillis();
        System.out.println("Hellinger: " + (endHellingerTime - initHellingerTime));

        List<DiscreteVariable> classVars = initialStructure.getLatentVariables();
        List<DiscreteVariable> featureVars = initialStructure.getManifestVariables();

        /* 2 - Crear la lista de arcos prohibidos (no se pueden añadir arcos entre MVs) */
        Map<DiscreteVariable, List<DiscreteVariable>> extraForbiddenAddArcs = new HashMap<>();
        for(DiscreteVariable featureVar: featureVars)
            extraForbiddenAddArcs.put(featureVar, new ArrayList<>(initialStructure.getManifestVariables())); // Lo pasamos como copia por si acaso, creo que no se modifica pero igual

        // Nota: Aquellos arcos que no se pueden añadir no pueden por tanto revertirse, por lo que no es necesario incluir codigo para ello.

        /* 2 - Crear el Structural EM */
        StructuralEM sem = new StructuralEM(classVars, featureVars,
                new HashMap<>(),
                extraForbiddenAddArcs, // Son los arcos que no se pueden añadir ADEMAS de los propios de una estructura MBC
                new HashMap<>(),
                new DagStructure(),
                em,
                Integer.MAX_VALUE, // Max iterations
                Integer.MAX_VALUE); // Max number of parents

        /* 3 - Crear los operadores del latent Hill-climber con SEM */
        Set<LatentMbcHcOperator> latentMbcHcOperators = new LinkedHashSet<>();
        latentMbcHcOperators.add(new IncreaseLatentCardinality(maxLatentVarCardinality)); // Le pasamos la cardinalidad maxima de una LV
        latentMbcHcOperators.add(new DecreaseLatentCardinality(minLatentVarCardinality)); // Le pasamos la cardinalidad minima de una LV

        /* 4 - Crear el MBC Hill-climber con SEM */
        LatentMbcHcWithSEM latentMbcHillClimbingWithSEM = new LatentMbcHcWithSEM(Integer.MAX_VALUE, threshold, latentMbcHcOperators);

        /* 5 - Aprender el modelo latente MBC */
        return latentMbcHillClimbingWithSEM.learnModel(initialStructure, data, sem);
    }

    /**
     * Version "RAPIDA" donde se omite unicamente el operador de añadir nodos, el cual es el mas costoso de todos, ya que
     * por cada par de MVs que considera necesita ejecutar el Structural EM, lo cual es muy costoso de ejecutar.
     */
    public static LearningResult<DiscreteBayesNet> learnFastModelWithHellingerAndRemoveNode(DiscreteData data,
                                                                                            HellingerFinder hellingerFinder,
                                                                                            double threshold,
                                                                                            AbstractEM em,
                                                                                            int maxLatentVarCardinality,
                                                                                            int minLatentVarCardinality) {

        /* 1 - Generamos la estructura inicial mediante el HellingerFinder */
        double initHellingerTime = System.currentTimeMillis();
        DiscreteBayesNet initialStructure = hellingerFinder.find(data, em);
        double endHellingerTime = System.currentTimeMillis();
        System.out.println("Hellinger: " + (endHellingerTime - initHellingerTime));

        List<DiscreteVariable> classVars = initialStructure.getLatentVariables();
        List<DiscreteVariable> featureVars = initialStructure.getManifestVariables();

        /* 2 - Crear la lista de arcos prohibidos (no se pueden añadir arcos entre MVs) */
        Map<DiscreteVariable, List<DiscreteVariable>> extraForbiddenAddArcs = new HashMap<>();
        for(DiscreteVariable featureVar: featureVars)
            extraForbiddenAddArcs.put(featureVar, new ArrayList<>(initialStructure.getManifestVariables())); // Lo pasamos como copia por si acaso, creo que no se modifica pero igual

        // Nota: Aquellos arcos que no se pueden añadir no pueden por tanto revertirse, por lo que no es necesario incluir codigo para ello.

        /* 2 - Crear el Structural EM */
        StructuralEM sem = new StructuralEM(classVars, featureVars,
                new HashMap<>(),
                extraForbiddenAddArcs, // Son los arcos que no se pueden añadir ADEMAS de los propios de una estructura MBC
                new HashMap<>(),
                new DagStructure(),
                em,
                Integer.MAX_VALUE, // Max iterations
                Integer.MAX_VALUE); // Max number of parents

        /* 3 - Crear los operadores del latent Hill-climber con SEM */
        Set<LatentMbcHcOperator> latentMbcHcOperators = new LinkedHashSet<>();
        latentMbcHcOperators.add(new IncreaseLatentCardinality(maxLatentVarCardinality)); // Le pasamos la cardinalidad maxima de una LV
        latentMbcHcOperators.add(new DecreaseLatentCardinality(minLatentVarCardinality)); // Le pasamos la cardinalidad minima de una LV
        latentMbcHcOperators.add(new RemoveLatentNode());

        /* 4 - Crear el MBC Hill-climber con SEM */
        LatentMbcHcWithSEM latentMbcHillClimbingWithSEM = new LatentMbcHcWithSEM(Integer.MAX_VALUE, threshold, latentMbcHcOperators);

        /* 5 - Aprender el modelo latente MBC */
        return latentMbcHillClimbingWithSEM.learnModel(initialStructure, data, sem);
    }
}
