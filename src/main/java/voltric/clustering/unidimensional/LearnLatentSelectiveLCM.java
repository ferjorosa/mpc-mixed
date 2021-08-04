package voltric.clustering.unidimensional;

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

/**
 * Codigo inspirado por el articulo de Barash & Friedman (2002).
 *
 * No todas las MVs del modelo tienen porque ser hijas de la variable de clustering en los selective LCMs.
 *
 * NO tiene arcos dependientes del contexto.
 */
public class LearnLatentSelectiveLCM {

    public static LearningResult<DiscreteBayesNet> learnModel(int cardinality,
                                                              DiscreteData data,
                                                              AbstractEM em) {
        /* 1 - Creamos el LCM inicial con la cardinalidad establecida */
        HLCM initialModel = HlcmCreator.createLCM(data.getVariables(), cardinality, new Random());

        /* 2 - Aprendemos los parametros como inicializacion del SEM */
        LearningResult<DiscreteBayesNet> initialModelResult = em.learnModel(initialModel, data);
        initialModel = (HLCM) initialModelResult.getBayesianNetwork();

        /* 3 - Aplicamos el SEM */
        List<DiscreteVariable> classVars = initialModel.getLatentVariables();
        List<DiscreteVariable> featureVars = initialModel.getManifestVariables();

        /* 3.1 - Crear la lista de arcos prohibidos a añadir (no se pueden añadir arcos entre MVs) */
        Map<DiscreteVariable, List<DiscreteVariable>> extraForbiddenAddArcs = new HashMap<>();
        for(DiscreteVariable featureVar: featureVars)
            extraForbiddenAddArcs.put(featureVar, new ArrayList<>(initialModel.getManifestVariables())); // Lo pasamos como copia por si acaso, creo que no se modifica pero igual

        StructuralEM sem = new StructuralEM(classVars, featureVars,
                new HashMap<>(),
                extraForbiddenAddArcs,
                new HashMap<>(),
                new DagStructure(),
                em,
                Integer.MAX_VALUE,
                Integer.MAX_VALUE);

        return sem.learnModel(initialModelResult, data);
    }
}
