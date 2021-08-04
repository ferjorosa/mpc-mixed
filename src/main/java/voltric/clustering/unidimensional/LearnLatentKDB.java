package voltric.clustering.unidimensional;

import voltric.clustering.multidimensional.mbc.LatentMbcHcWithSEM;
import voltric.clustering.multidimensional.mbc.operator.IncreaseLatentCardinality;
import voltric.clustering.multidimensional.mbc.operator.LatentMbcHcOperator;
import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.learning.structure.latent.StructuralEM;
import voltric.learning.structure.type.DagStructure;
import voltric.model.DiscreteBayesNet;
import voltric.model.HLCM;
import voltric.model.creator.HlcmCreator;
import voltric.variables.DiscreteVariable;

import java.util.*;

public class LearnLatentKDB {

    public static LearningResult<DiscreteBayesNet> learnModel(int cardinality,
                                                              DiscreteData data,
                                                              AbstractEM em){

        // Create and learn the initial LCM
        HLCM initialLcm = HlcmCreator.createLCM(data.getVariables(), cardinality, new Random());

        return learnModel(initialLcm, data, em);
    }

    public static LearningResult<DiscreteBayesNet> learnModel(HLCM initialModel,
                                                              DiscreteData data,
                                                              AbstractEM em){

        /* 1 - Clonamos el modelo para no modificar el que se nos pasa */
        HLCM clonedInitialModel = initialModel.clone();

        /* 2 - Aprendemos los parametros como inicializacion del SEM */
        LearningResult<DiscreteBayesNet> initialModelResult = em.learnModel(clonedInitialModel, data);

        /* 3 - Aplicamos el SEM */
        List<DiscreteVariable> classVars = clonedInitialModel.getLatentVariables();
        List<DiscreteVariable> featureVars = clonedInitialModel.getManifestVariables();

        // Impedimos que se borre la estructura Class-Bridge
        Map<DiscreteVariable, List<DiscreteVariable>> forbiddenDeleteArcs = new HashMap<>();
        for(DiscreteVariable classVar: classVars)
            forbiddenDeleteArcs.put(classVar, clonedInitialModel.getVariables());


        StructuralEM sem = new StructuralEM(classVars, featureVars,
                forbiddenDeleteArcs,
                new HashMap<>(),
                new HashMap<>(),
                new DagStructure(),
                em,
                Integer.MAX_VALUE,
                Integer.MAX_VALUE);

        return sem.learnModel(initialModelResult, data);
    }

    public static LearningResult<DiscreteBayesNet> learnModelToMaxCardinality(HLCM initialModel,
                                                                              DiscreteData data,
                                                                              AbstractEM em,
                                                                              double threshold,
                                                                              int maxCardinality) {

        /* 1 - Crear la lista de arcos prohibidos a añadir (no se pueden añadir arcos entre MVs) para el SEM */
        List<DiscreteVariable> classVars = initialModel.getLatentVariables();
        List<DiscreteVariable> featureVars = initialModel.getManifestVariables();

        /* 2 - Impedimos que se borre la estructura Class-Bridge */
        Map<DiscreteVariable, List<DiscreteVariable>> forbiddenDeleteArcs = new HashMap<>();
        for(DiscreteVariable classVar: classVars)
            forbiddenDeleteArcs.put(classVar, initialModel.getVariables());

        /* 3 - Crear el Structural EM */
        StructuralEM sem = new StructuralEM(classVars, featureVars,
                forbiddenDeleteArcs,
                new HashMap<>(),
                new HashMap<>(),
                new DagStructure(),
                em,
                Integer.MAX_VALUE,
                Integer.MAX_VALUE);

        /* 4 - Creamos un Hill-climber donde el unico operador posible es incrementar la cardinalidad + SEM */
        Set<LatentMbcHcOperator> latentMbcHcOperators = new LinkedHashSet<>();
        latentMbcHcOperators.add(new IncreaseLatentCardinality(maxCardinality)); // Le pasamos la cardinalidad maxima de una LV
        LatentMbcHcWithSEM latentMbcHillClimbingWithSEM = new LatentMbcHcWithSEM(Integer.MAX_VALUE, threshold, latentMbcHcOperators);

        /* 5 - Aprender el modelo latente MBC (en este caso es unidimensional) */
        return latentMbcHillClimbingWithSEM.learnModel(initialModel, data, sem);
    }
}
