package voltric.inference;

import voltric.data.DiscreteData;
import voltric.graph.AbstractNode;
import voltric.graph.DirectedNode;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.potential.Function;
import voltric.variables.DiscreteVariable;
import voltric.variables.Variable;

import java.util.*;

/*
    Clase diseñada para hacer inferencia exacta con un subTree especifico. Se va a utilizar en LocalEM. Si bien podria
    utilizar la clase de CliqueTreePropagation, prefiero separar ambos conceptos para poder cambiar cosas sin romper lo
    que ya se encuentra implementado.

    Dos estrategias posibles para poder realizar inferencia sobre un subTree (local) y quedarnos con un score global:

    1) La primera estrategia esta inspirada en el paso de mensajes tradicional y por cada propagacion se computa la LL
       de todos los nodos, incluso de aquellos que no se ven afectados durante la inferencia. Es como lo tengo actualmente
       hecho en VMP, pero puede hacerse mas eficiente si nos aprovechamos de la descomponibilidad del score. Los pasos son:
        - collect messages (from All)
        - distribute messages (to focus)
        - computeLL (All)

    2) La segunda estrategia es mas sofisticada en el hecho de que hace uso de la localidad del score. La idea es crear
       un clique Tree, delimitar un subTree donde se va a realizar la inferencia y propagar unicamente por los nodos de
       dicho subarbol. Dado que en este caso no estamos estimando el score global, una vez hayamos finalizado el aprendizaje
       local, debemos simplemente delimitar el subTree como el total del arbol y propagar una vez para obtener el score
       de todos los nodos (no solo aquellos que habian sido seleccionados inicialmente).
        - subtree (focus)
        - collect (focus)
        - distribute (focus)
        - computeScore (focus)
        - ... LocalEM until convergence
        - score (all)

    3) Hay una tercera opcion que es la que creo que hace Poon y compañia que es propagar por el subarbol podado pero
       copiar mensajes de las partes del arbol no afectadas para poder computar la LL completa.

 */

// Siempre se necesita tener el mensaje del vecino al nodo en cuestion.
//      En el caso de la likelihood, necesitas el mensaje de los vecinos del pivot al pivot
//      En el caso de familyBelief, necesitas el mensaje de los vecinos del familyClique al familyClique
//
// Conclusiones de esto:
//      Esto me da a entender que necesitas como minimo una pasada de recoleccion para el likelihood full
//      Esto me dice que el pivot deberia claramente ser uno de los nodos principales de la propagacion local.
// Nota sobre Likelihood y absorbEvidence
// Para calcular el likelihood completo es necesario tener los potenciales de todos los nodos (absorbEvidence)
// Se podria optimizar para solo considerar los vecinos de los nodos del subTree y hacer scores locales, pero requeriria
// mas trabajo para hacerlo eficiente, ya que al final del LocalEM es interesante tener el likelihood completo.
public class LocalCliqueTreePropagation {

    /** The BN under query */
    private DiscreteBayesNet bayesNet;

    /** The associated Clique Tree */
    private CliqueTree cliqueTree;

    /** Evidence values used in the inference */
    private Map<DiscreteVariable, Integer> evidence;

    /** Variables whose cliques are going to be considered in the inference process. */
    Set<DiscreteVariable> localVariables;

    /** Main constructor */
    public LocalCliqueTreePropagation(DiscreteBayesNet bayesNet, Set<DiscreteVariable> localVariables) {
        this.bayesNet = bayesNet;
        this.cliqueTree = new CliqueTree(this.bayesNet);
        this.evidence = new HashMap<>();
        setFocusedSubTree(localVariables);
    }

    /** Sets the focused subTree where propagation is going to be done */
    public void setFocusedSubTree(Set<DiscreteVariable> localVariables) {

        this.localVariables = localVariables;

        /* Set which nodes are going to be affected by propagation */
        Set<CliqueNode> localNodes = new LinkedHashSet<>();
        for(DiscreteVariable var: localVariables){
            CliqueNode localNode = cliqueTree.getFamilyClique(var);
            localNodes.add(localNode);
        }

        this.cliqueTree.setFocusedSubtree(localNodes);

        /* Sets the smallest clique of the focused subtree as pivot (for fast likelihood computation) */
        int minCard = Integer.MAX_VALUE;
        for (CliqueNode node: this.cliqueTree._focusedSubtree) {
            int card = node.getCardinality();
            if (card < minCard) {
                minCard = card;
                setPivot(node);
            }
        }
    }

    /** Sets the pivot for propagation */
    public void setPivot(CliqueNode node) { this.cliqueTree.setPivot(node); }

    /** Propagates evidence through the model */
    public double propagate() {

        /* Distributes the evidence through the Clique Tree */
        absorbEvidenceV2();

        CliqueNode pivot = cliqueTree.getPivot();

        /* Collects messages from neighbors of the pivot. Only considers clique nodes of local variables */
        for (AbstractNode<String> neighbor : pivot.getNeighbors())
            collectMessage((CliqueNode) neighbor, pivot);

        /* Distributes messages to neighbors of pivot Only considers clique nodes of local variables */
        for (AbstractNode<String> neighbor : pivot.getNeighbors())
            distributeMessage(pivot, (CliqueNode) neighbor);

        return computeLikelihood();
    }

    /** Returns the posterior probability distribution of the specified variable's family (no matter if they are hidden or observed) */
    public Function computeFamilyBelief(DiscreteVariable var) {
        if(var == null)
            throw new IllegalArgumentException("Variable cannot be null");
        if(!bayesNet.containsVar(var))
            throw new IllegalArgumentException("The variable under query is not present in the model");

        /* Hidden and observed variables. Hidden means without evidence, it doesnt have to be a latent variable. */
        LinkedList<DiscreteVariable> hdnVars = new LinkedList<>();
        ArrayList<DiscreteVariable> obsVars = new ArrayList<>();
        ArrayList<Integer> obsVals = new ArrayList<>();

        /* Distinguish if the main var is hidden or observed */
        if (evidence.containsKey(var)) {
            obsVars.add(var);
            obsVals.add(evidence.get(var));
        } else
            hdnVars.add(var);

        /* Distinguish if parents of the main var are hidden or observed */
        DiscreteBeliefNode node = bayesNet.getNode(var);
        for (AbstractNode parent : node.getParents()) {
            DiscreteBeliefNode bParent = (DiscreteBeliefNode) parent;
            DiscreteVariable vParent = bParent.getVariable();

            if (evidence.containsKey(vParent)) {
                obsVars.add(vParent);
                obsVals.add(evidence.get(vParent));
            } else
                hdnVars.add(vParent);
        }

        /* Belief over observed variables. If there are no hiddne variables, we return this function */
        Function obsBel = Function.createIndicatorFunction(obsVars, obsVals);
        if (hdnVars.isEmpty())
            return obsBel;

        // belief over hidden variables
        Function hdnBel = Function.createIdentityFunction();

        // computes potential at family covering clique
        CliqueNode familyClique = cliqueTree.getFamilyClique(var);

        // times up functions attached to family covering clique
        for (Function function : familyClique.getFunctions())
            hdnBel = hdnBel.times(function);

        // (In the HLCM propogation case) After this, the hdnBel is superior to
        // any funtion multiplied.
        for (AbstractNode neighbor : familyClique.getNeighbors()) {
            hdnBel = hdnBel.times(((CliqueNode) neighbor).getMessageTo(familyClique));
        }

        if (!(hdnVars.size() == hdnBel.getDimension())) {
            // marginalizes potential
            hdnBel = hdnBel.marginalize(hdnVars);
        }

        // normalizes potential
        hdnBel.normalize();

        return hdnBel.times(obsBel);
    }

    /** Assigns evidence to the Clique Tree */
    public void setEvidence(Map<DiscreteVariable, Integer> evidenceValues) {
        clearEvidence();

        for (DiscreteVariable var: evidenceValues.keySet()) {
            /* Ignore variables with missing values */
            if (evidenceValues.get(var) == DiscreteData.MISSING_VALUE)
                continue;
            /* Throw an exception if variable doesn't belong to the BN */
            if(!bayesNet.containsVar(var))
                throw new IllegalArgumentException("The Bayes net does not contain the variable: " + var.getName());
            /* Throw and exception if evidence value is incorrect */
            int value = evidenceValues.get(var);
            if(!var.isValuePermitted(value))
                throw new IllegalArgumentException("the state [" + value + "] is not valid for the variable: " + var.getName());

            evidence.put(var, value);
        }
    }

    /** Clears the evidence entered into this inference engine. */
    public void clearEvidence() { evidence.clear(); }

    public DiscreteBayesNet getBayesNet() { return this.bayesNet; }

    /** Uses the evidence to prepare the CT for propagation. Its an internal step of propagate(). */
    private void absorbEvidence() {

        /* Reset clique nodes */
        for(AbstractNode<String> node: cliqueTree.getNodes()){
            CliqueNode cliqueNode = (CliqueNode) node;
            cliqueNode.clearFunctions();
            cliqueNode.clearQualifiedNeiMsgs();
            cliqueNode.setMsgsProd(Function.createIdentityFunction());
        }

        /* Initialize functions. Only consider functions from local variables */
        LinkedHashMap<Variable, Function> functions = new LinkedHashMap<>();
        for(DiscreteVariable var: localVariables)
            functions.put(var, bayesNet.getNode(var).getCpt());

        /* Project functions whose variables are in the evidence (also, project its children nodes with its parent evidence) */
        for(DiscreteVariable var: evidence.keySet()){
            int value = evidence.get(var);
            DiscreteBeliefNode bNode = bayesNet.getNode(var);
            functions.put(var, functions.get(var).project(var, value));
            for(DirectedNode<Variable> child: bNode.getChildren()){
                Variable childVar = child.getContent();
                functions.put(childVar, functions.get(childVar).project(var, value));
            }
        }

        /* Attach function to family covering clique */
        for(DiscreteVariable var: localVariables)
            cliqueTree.getFamilyClique(var).attachFunction(functions.get(var));
    }

    // Creo esta version porque en localEM se calcula el likelihood completo, y para ello es necesario tener los potenciales de todos los nodos
    // Se podria optimizar para solo considerar los vecinos de los nodos del subTree y hacer scores locales, pero requeriria
    // mas trabajo para hacerlo eficiente
    private void absorbEvidenceV2() {

        /* Reset clique nodes */
        for(AbstractNode<String> node: cliqueTree.getNodes()){
            CliqueNode cliqueNode = (CliqueNode) node;
            cliqueNode.clearFunctions();
            cliqueNode.clearQualifiedNeiMsgs();
            cliqueNode.setMsgsProd(Function.createIdentityFunction());
        }

        /* Initialize functions. Only consider functions from local variables */
        LinkedHashMap<Variable, Function> functions = new LinkedHashMap<>();
        for(DiscreteBeliefNode node: bayesNet.getNodes())
            functions.put(node.getVariable(), node.getCpt());

        /* Project functions whose variables are in the evidence (also, project its children nodes with its parent evidence) */
        for(DiscreteVariable var: evidence.keySet()){
            int value = evidence.get(var);
            DiscreteBeliefNode bNode = bayesNet.getNode(var);
            functions.put(var, functions.get(var).project(var, value));
            for(DirectedNode<Variable> child: bNode.getChildren()){
                Variable childVar = child.getContent();
                functions.put(childVar, functions.get(childVar).project(var, value));
            }
        }

        /* Attach function to family covering clique */
        for(DiscreteVariable var: bayesNet.getVariables())
            cliqueTree.getFamilyClique(var).attachFunction(functions.get(var));
    }

    /** Collects messages around the source and sends an aggregated message to the destination. */
    private void collectMessage(CliqueNode source, CliqueNode destination) {
        if (source.getMessageTo(destination) == null || cliqueTree.inFocusedSubtree(source)) {
            // collects messages from neighbors of source except destination
            for (AbstractNode<String> neighbor : source.getNeighbors()) {
                if (neighbor != destination)
                    collectMessage((CliqueNode) neighbor, source);
            }
            sendMessage(source, destination);
        }
    }

    /** Sends an aggregated message from the source to the destination and distributes the message around the destination. */
    private void distributeMessage(CliqueNode source, CliqueNode destination) {
        if (cliqueTree.inFocusedSubtree(destination)) {
            sendMessage(source, destination);
            // distributes messages to neighbors of destination except source
            for (AbstractNode<String> neighbor : destination.getNeighbors()) {
                if (neighbor != source)
                    distributeMessage(destination, (CliqueNode) neighbor);
            }
        }
    }

    /** Sends a message from the source to the destiation. */
    private void sendMessage(CliqueNode source, CliqueNode destination) {
        Function message = Function.createIdentityFunction();
        double normalization = 1.0;
        double logNormalization = 0;

        for (AbstractNode<String> neighbor : source.getNeighbors()) {
            if (neighbor != destination) {
                CliqueNode clique = (CliqueNode) neighbor;
                message = message.times(clique.getMessageTo(source));
                normalization *= clique.getNormalizationTo(source);
                logNormalization += clique.getLogNormalizationTo(source);
            }
        }

        for (Function function : source.getFunctions()) {
            message = message.times(function);
        }

        // sums out difference between source and destination
        for (DiscreteVariable var : source.getDifferenceTo(destination)) {
            if (!evidence.containsKey(var)) {
                message = message.sumOut(var);
            }
        }

        // normalizes to alleviate round off error
        double n = message.normalize();
        normalization *= n;
        logNormalization += Math.log(n); //TODO: Utils.log()?

        // TODO: 18-09-2019: Lo he comentado para ver que resultados daba
        //assert normalization >= Double.MIN_NORMAL;
        //if(normalization < Double.MIN_NORMAL)
        //    throw new IllegalStateException("normalization value lower than Double.MIN_NORMAL");

        // saves message and normalization
        source.setMessageTo(destination, message);
        source.setNormalizationTo(destination, normalization);
        source.setLogNormalizationTo(destination, logNormalization);
    }

    /** Returns the likelihood of the evidences on the associated BN. Requires propagation. */
    private double computeLikelihood() {
        CliqueNode pivot = cliqueTree.getPivot();

        // times up functions attached to pivot
        Function potential = Function.createIdentityFunction();
        for (Function function : pivot.getFunctions()) {
            potential = potential.times(function);
        }

        // times up messages to pivot
        double normalization = 1.0;
        double logNormalization = 0;
        for (AbstractNode<String> neighbor : pivot.getNeighbors()) {
            CliqueNode clique = (CliqueNode) neighbor;
            potential = potential.times(clique.getMessageTo(pivot));
            normalization *= clique.getNormalizationTo(pivot);
            logNormalization += clique.getLogNormalizationTo(pivot);
        }

        double n = potential.sumUp();
        double ll = logNormalization + Math.log(n); // Log-likelihood
        return n * normalization;
    }
}
