package voltric.learning.structure.latent;

import voltric.clustering.util.GenerateCompleteData;
import voltric.data.DiscreteData;
import voltric.learning.LearningResult;
import voltric.learning.parameter.em.AbstractEM;
import voltric.learning.score.ScoreType;
import voltric.learning.structure.hillclimbing.local.*;
import voltric.learning.structure.type.StructureType;
import voltric.model.DiscreteBayesNet;
import voltric.variables.DiscreteVariable;

import java.util.*;

/**
 * Hay que crear:
 *
 * Un constructor con:
 * - El objeto EM
 * - El tipo de estructura que se va a aprender: Class-Bridge y Feature
 * - Variables de clase
 * - Variables atributo
 *
 * Ademas es necesario:
 * - Establecer que variables son clase y cuales son feature (no vale con latent y manifest porque se van a completar los datos)
 * - Generar 2 clases: CBStructure y FeatureStructure con 5 metodos cada uno
 */
// TODO: Hay que reestructurar la clase para que pueda ser reutilizable el SEM, pero que cada llamada solo dependa de la estrucutra en cuestion
    // TODO: Por ahora esta clase esta pensada unicamente para MBCs, se podria hacer una version general y otra MBC
public class StructuralEM {

    private List<DiscreteVariable> classVars;

    private List<DiscreteVariable> featureVars;

    // Realmente solo puede ser Forest, PolyForest y DAG, leer explicacion en el package "voltric.learning.structure.type"
    private StructureType mbcStructure;

    private AbstractEM em;

    private int maxIterations;

    private int maxNumberOfParents;

    private LocalHillClimbing hillClimbing;

    public StructuralEM(List<DiscreteVariable> classVars,
                        List<DiscreteVariable> featureVars,
                        StructureType mbcStructure,
                        AbstractEM em,
                        int maxIterations,
                        int maxNumberOfParents) {
        this.classVars = classVars;
        this.featureVars = featureVars;
        this.mbcStructure = mbcStructure;
        this.em = em;
        this.maxIterations = maxIterations;
        this.maxNumberOfParents = maxNumberOfParents;

        // Creamos el objeto de hillClimbing que servira para obtener la estructura adecuada con los datos completoss
        this.hillClimbing = generateMbcHillClimbing(new HashMap<>(), new HashMap<>(), new HashMap<>());
    }

    // Los extraForbiddenArcs se añaden a los propios establecidos por las restricciones de la estructura MBC
    // Dado que no impedimos ninguna eliminacion de arcos, forbiddenDeleteArcs no es "extra"
    public StructuralEM(List<DiscreteVariable> classVars,
                        List<DiscreteVariable> featureVars,
                        Map<DiscreteVariable, List<DiscreteVariable>> forbiddenDeleteArcs,
                        Map<DiscreteVariable, List<DiscreteVariable>> extraForbiddenAddArcs,
                        Map<DiscreteVariable, List<DiscreteVariable>> extraForbiddenReverseArcs,
                        StructureType mbcStructure,
                        AbstractEM em,
                        int maxIterations,
                        int maxNumberOfParents) {
        this.classVars = classVars;
        this.featureVars = featureVars;
        this.mbcStructure = mbcStructure;
        this.em = em;
        this.maxIterations = maxIterations;
        this.maxNumberOfParents = maxNumberOfParents;

        // Creamos el objeto de hillClimbing que servira para obtener la estructura adecuada con los datos completoss
        this.hillClimbing = generateMbcHillClimbing(forbiddenDeleteArcs, extraForbiddenAddArcs, extraForbiddenReverseArcs);
    }

    // el modelo seedNetResult debe de encontrarse aprendido con los parametros adecuados (EM algorithm)
    public LearningResult<DiscreteBayesNet> learnModel(LearningResult<DiscreteBayesNet> seedNetResult, DiscreteData data) {

        DiscreteBayesNet currentNet = seedNetResult.getBayesianNetwork();
        double currentScore = seedNetResult.getScoreValue();

        int iterations = 0;
        while(iterations < this.maxIterations) {

            iterations++;

            System.out.println("SEM iteration "+ iterations);

            // Generamos los datos completos a partir de los parametros de la currentNet
            DiscreteData completeData = GenerateCompleteData.generateMultidimensional(data, currentNet);

            // Aprendemos la nueva estructura S(n+1) a partir de los datos completados con el modelo antiguo [S(n),O(n)]
            DiscreteBayesNet hcNet = this.hillClimbing.learnModel(currentNet, completeData).getBayesianNetwork();

            // Una vez tenemos la estructura S(n+1), calculamos los parametros O(n+1)
            LearningResult<DiscreteBayesNet> iterationResult = this.em.learnModel(hcNet, data);

            // Si el nuevo modelo [S(n+1),O(n+1)] no supera el score del actual, se devuelve [S(n),O(n)]
            if(currentScore >= iterationResult.getScoreValue() || Math.abs(iterationResult.getScoreValue() - currentScore) < this.em.getThreshold())
                return new LearningResult<>(currentNet, currentScore, this.em.getScoreType());

            // En caso contrario, se actualiza el modelo con el nuevo
            currentNet = iterationResult.getBayesianNetwork();
            currentScore = iterationResult.getScoreValue();
        }

        return new LearningResult<>(currentNet, currentScore, em.getScoreType());
    }

    // Las unicas restricciones que contiene son las propias de la estructura MBC
    private LocalHillClimbing generateMbcHillClimbing(Map<DiscreteVariable, List<DiscreteVariable>> forbiddenDeleteArcs,
                                                      Map<DiscreteVariable, List<DiscreteVariable>> extraForbiddenAddArcs,
                                                      Map<DiscreteVariable, List<DiscreteVariable>> extraForbiddenReverseArcs) {

        Set<LocalHcOperator> operators = new LinkedHashSet<>();

        /* Definimos el conjunto de arcos prohibidos a ser añadidos (lo propio de las estructuras MBC + el extra del usuario) */
        Map<DiscreteVariable, List<DiscreteVariable>> forbiddenAddArcs = new HashMap<>();
        for(DiscreteVariable featureVar: this.featureVars)
            forbiddenAddArcs.put(featureVar, new ArrayList<>(this.classVars)); // Importante pasarle una copia

        /* Extra: */
        for(DiscreteVariable variable: extraForbiddenAddArcs.keySet())
            if(forbiddenAddArcs.containsKey(variable))
                forbiddenAddArcs.get(variable).addAll(extraForbiddenAddArcs.get(variable));
            else
                forbiddenAddArcs.put(variable, extraForbiddenAddArcs.get(variable));

        /* Definimos el conjunto de arcos prohibidos a ser revertidos (lo propio de las estructuras MBC + el extra del usuario) */
        Map<DiscreteVariable, List<DiscreteVariable>> forbiddenReverseArcs = new HashMap<>();
        for(DiscreteVariable classVar: this.classVars)
            forbiddenReverseArcs.put(classVar, new ArrayList<>(this.featureVars)); // Importante pasarle una copia
        /* Extra: */
        for(DiscreteVariable variable: extraForbiddenReverseArcs.keySet())
            if(forbiddenReverseArcs.containsKey(variable))
                forbiddenReverseArcs.get(variable).addAll(extraForbiddenReverseArcs.get(variable));
            else
                forbiddenReverseArcs.put(variable, extraForbiddenReverseArcs.get(variable));

        operators.add(new LocalAddArc(new ArrayList<>(), forbiddenAddArcs, this.maxNumberOfParents));
        operators.add(new LocalDeleteArc(new ArrayList<>(), forbiddenDeleteArcs));
        operators.add(new LocalReverseArc(new ArrayList<>(), forbiddenReverseArcs, this.maxNumberOfParents));

        /* Utilizamos el threshold y score type del EM */
        return new LocalHillClimbing(operators, this.maxIterations, this.em.getThreshold(), this.em.getScoreType(), this.mbcStructure);
    }

    /**
     * Instancia del algoritmo EM para poder ejecutar el SEM con un punto de inicio de calidad
     *
     * TODO: Nota: Lo he puesto deprecated para acordarme de que no es la mejor manera de hacerlo, lo mejor seria que estuviese
     * concentrado el paso inicial del EM en el SEM, pero se que tengo codigo que lo ha ejecutado el EM con anterioridad
     * y para no repetir lo separe, pero no tiene sentido.
     *
     * @return
     */
    @Deprecated
    public AbstractEM getEm() {
        return em;
    }

    public ScoreType getScoreType() {
        return this.em.getScoreType();
    }

    // Pensado originalmente para operadores del tipo "AddLatentNode"
    public void addLatentVar(DiscreteVariable latentVar){
        this.classVars.add(latentVar);

        for(DiscreteVariable featureVar: this.featureVars){
            this.hillClimbing.newAddArcRestriction(featureVar, latentVar);
            this.hillClimbing.newReverseArcRestriction(latentVar, featureVar);
            // NOTA: No tiene sentido añadir restricciones de eliminacion de arcos, a menos que asi se especifique en la estructura inicial
        }
    }

    // Valido para "AddLatentNode", "IncreaseCard", "DecreaseCard" o "RemoveLatentNode"
    public void removeLatentVar(DiscreteVariable latentVar){
        this.classVars.remove(latentVar);

        this.hillClimbing.removeAddArcRestrictions(latentVar);
        this.hillClimbing.removeReverseArcRestrictions(latentVar);
    }

    // Pensado originalmente para "IncreaseLatentCard" y "DecreaseLatentCard", ya que se copian las restricciones de una LV existente
    public void addLatentVarWithRestrictions(DiscreteVariable newLatentVar, DiscreteVariable existingLatentVar) {
        this.classVars.add(newLatentVar);

        this.hillClimbing.copyArcRestrictions(existingLatentVar, newLatentVar);
    }
}
