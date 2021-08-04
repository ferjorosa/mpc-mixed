package voltric.model;

import org.apache.commons.lang3.builder.HashCodeBuilder;
import voltric.data.DiscreteData;
import voltric.graph.DirectedAcyclicGraph;
import voltric.graph.Edge;
import voltric.graph.TopologicalSorter;
import voltric.graph.UndirectedGraph;
import voltric.potential.Function;
import voltric.variables.DiscreteVariable;
import voltric.variables.Variable;
import voltric.variables.modelTypes.VariableType;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

/**
 * This class implements a Bayesian network where all its nodes are discrete.
 *
 * TODO: He eliminado el HashMap de DiscreteVariable -> DiscreteBeliefNode. si noto mucho overheat por los explicit castings lo vuelvo a poner y pista.
 *
 * TODO: Revisar que metodos realmente son propios de DiscreteBayesNet y cuales de Abstract, ya que se ha eliminado las LL
 */
public class DiscreteBayesNet extends AbstractBayesNet<DiscreteBeliefNode, DiscreteBeliefNode> {

    /**
     * Default constructor. It creates an empty Bayesian network with a default name.
     */
    public DiscreteBayesNet(){
        super("Discrete_Bayesian_network");
    }

    /**
     * Constructs a discrete Bayesian network with the argument name.
     *
     * @param name the network's name.
     */
    public DiscreteBayesNet(String name){
        super(name.trim());
    }

    /**
     * Copy constructor. The name, DAG and log-likelihoods are added to this new BayesNet. However, they are not added as
     * references, but as new instances.
     *
     * @param bayesNet the bayesNet being copied.
     */
    public DiscreteBayesNet(DiscreteBayesNet bayesNet){

        this.name = bayesNet.name;
        this.dag = new DirectedAcyclicGraph<>();

        // Iterate through the variables of the BN and create the corresponding nodes
        for(DiscreteVariable variable: bayesNet.getVariables())
            this.addNode(variable);

        // Iterate throught the edges of the BN and copy them
        for(Edge<Variable> edge: bayesNet.getEdges())
            this.addEdge(this.getNode(edge.getHead().getContent()), this.getNode(edge.getTail().getContent()));

        // Assign the corresponding CPTs to the new nodes of the BN
        // We need to first construct the structure and then assign the CPTs or an exception will be thrown
        for(DiscreteBeliefNode node: bayesNet.getNodes()) {
            Function clonedCPT = node.getCpt().clone();
            this.getNode(node.getVariable()).setCpt(clonedCPT);
        }
    }

    public final void initializeEmpty(DiscreteData data) {
        for(DiscreteVariable var: data.getVariables()) {

            // Add a new node to the BN
            this.addNode(var);

            // Create the new CPT for the node with its projected data
            Function newCpt = Function.createFunction(data.project(var));
            newCpt.normalize(var);
            this.getNode(var).setCpt(newCpt);
        }
    }

    /** {@inheritDoc} */
    @Override
    public DiscreteBeliefNode getNode(String name) {
        return (DiscreteBeliefNode) super.getNode(name);
    }

    /** {@inheritDoc} */
    @Override
    public DiscreteBeliefNode getNode(Variable variable) {
        return (DiscreteBeliefNode) super.getNode(variable);
    }

    public final List<DiscreteBeliefNode> getNodes(){
        return this.dag.getNodes().stream().map(x-> (DiscreteBeliefNode) x).collect(Collectors.toList());
    }

    /** {@inheritDoc} */
    @Override
    public List<DiscreteBeliefNode> getManifestNodes() {
        return this.getNodes().stream().filter(x->x.getVariable().isManifestVariable()).collect(Collectors.toList());
    }

    /** {@inheritDoc} */
    @Override
    public List<DiscreteBeliefNode> getLatentNodes() {
        return this.getNodes().stream().filter(x->x.getVariable().isLatentVariable()).collect(Collectors.toList());
    }

    /**
     * Creates a new {@link DiscreteBeliefNode} from the argument variable and adds it to the Bayesian network.
     *
     * @param variable variable used to create the new belief node.
     * @return the newly created belief node.
     */
    // TODO Por eso no quiero que este metodo sea polimorfico, ya que no tiene sentido al tener que utilizar instanceOf
    public DiscreteBeliefNode addNode(DiscreteVariable variable) {

        if(this.dag.containsNode(variable))
            throw new IllegalArgumentException("Node names must be unique.");

        // creates node
        DiscreteBeliefNode node = new DiscreteBeliefNode(this, variable);

        // Node added to the DAG
        this.dag.addNode(node);

        return node;
    }

    /**
     * Removes the specified node from this BN. This implementation extends
     * <code>AbstractGraph.removeNode(AbstractNode)</code> such that map from
     * variables to nodes will be updated and all loglikelihoods will be
     * expired.
     *
     * @param node node to be removed from this BN.
     */
    public final void removeNode(DiscreteBeliefNode node) {
        this.dag.removeNode(node);
    }

    /**
     * Returns the list of variables in this BN. For the sake of efficiency, this implementation returns the reference
     * to the private field.
     *
     * @return the list of variables in this BN.
     */
    public final List<DiscreteVariable> getVariables() {
        //return this.dag.getContents().keySet().stream().map(x-> (DiscreteVariable) x).collect(Collectors.toList());
        // He quitado getContents porque no mantiene el orden al llamar al keyset del hashMap
        return this.getNodes().stream().map(DiscreteBeliefNode::getVariable).collect(Collectors.toList());
    }

    /**
     * Returns the latent (hidden) variables of the model.
     *
     * @return the latent (hidden) variables of the model.
     */
    public final List<DiscreteVariable> getLatentVariables(){
        return this.getVariables().stream().filter(x-> x.isLatentVariable()).collect(Collectors.toList());
    }

    public final DiscreteVariable getLatentVariable(String name){
        List<DiscreteVariable> latentVars = this.getLatentVariables().stream().filter(x -> x.getName().equals(name)).collect(Collectors.toList());

        if (latentVars.size() > 1)
            throw new IllegalStateException("There shouldn't be two variables with the same name");

        return latentVars.get(0);
    }

    /**
     * Returns the manifest (observable) variables of the model.
     *
     * @return the manifest (observable) variables of the model.
     */
    public final List<DiscreteVariable> getManifestVariables(){
        return this.getVariables().stream().filter(x-> x.isManifestVariable()).collect(Collectors.toList());
    }

    /**
     * Returns {@code true} if all the variables of the collection belong to the network, {@code false} otherwise.
     *
     * @param vars the collection of variables.
     * @return {@code true} if all the variables of the collection belong to the network, {@code false} otherwise.
     */
    public final boolean containsVars(Collection<DiscreteVariable> vars) {
        return this.getVariables().containsAll(vars);
    }

    /**
     * Returns {@code true} if the specific var belongs to the network, {@code false} otherwise.
     *
     * @param var the argument variable.
     * @return {@code true} if the specific var belongs to the network, {@code false} otherwise.
     */
    public final boolean containsVar(DiscreteVariable var) {
        List<DiscreteVariable> variables = this.getVariables();
        return variables.contains(var);
    }

    /**
     * Returns the standard dimension, namely, the number of free parameters in the CPTs, of this BN.
     *
     * @return the standard dimension of this BN.
     */
    public final int computeDimension() {
        // sums up dimension for each node
        int dimension = 0;

        for (DiscreteBeliefNode node : this.getNodes()) {
            dimension += node.computeDimension();
        }

        return dimension;
    }

    /**
     * Returns the moral graph of its DAG.
     *
     * @return the moral graph of its DAG.
     */
    public final UndirectedGraph<Variable> computeMoralGraph(){
        return this.dag.computeMoralGraph();
    }

    public final List<DiscreteBeliefNode> topologicalSort() {
        return TopologicalSorter.sort(this.dag).stream().map(x->this.getNode(x.getContent())).collect(Collectors.toList());
    }

    /**
     * Randomly sets the parameters of this BN.
     */
    // kmpoon's TODO: avoid redundant operations on CPTs.
    public final void randomlyParameterize() {
        for (DiscreteBeliefNode node : this.getNodes()) {
            node.randomlyParameterize();
        }
    }

    public final void randomlyParameterize(Random random) {
        for (DiscreteBeliefNode node : this.getNodes()) {
            node.randomlyParameterize(random);
        }
    }

    public final void randomlyParameterize(Random random, Collection<DiscreteBeliefNode> mutableNodes) {
        if(!this.getNodes().containsAll(mutableNodes))
            throw new IllegalArgumentException("All the mutable nodes must be in this BN");

        for (DiscreteBeliefNode node : mutableNodes)
            node.randomlyParameterize(random);
    }

    /**
     * Randomly sets the parameters of the specified list of nodes in this BN.
     *
     * @param mutableNodes list of nodes whose parameters are to be randomized.
     */
    // kmpoon's TODO: avoid redundant operations on CPTs.
    public final void randomlyParameterize(Collection<DiscreteBeliefNode> mutableNodes) {
        // mutable nodes must be in this BN
        if(!this.getNodes().containsAll(mutableNodes))
            throw new IllegalArgumentException("All the mutable nodes must be in this BN");

        for (DiscreteBeliefNode node : mutableNodes) {
            node.randomlyParameterize();
        }
    }

    /**
     * Returns {@code true} if the object is a {@code DiscreteBayesNet} with equal fields (inherited ones included).
     *
     * @param object the object to test equality against.
     * @return true if {@code object} equals this.
     */
    @Override
    public boolean equals(Object object){
        if(this == object)
            return true;

        if(object.getClass() != this.getClass())
            return  false;

        DiscreteBayesNet bayesNet = (DiscreteBayesNet) object;
        return this.name.equals(bayesNet.name)
                && this.dag.equals(bayesNet.dag);
    }

    /**
     * Returns the object's hashcode.
     *
     * @return the object's hashcode.
     */
    @Override
    public int hashCode() {
        return new HashCodeBuilder(3, 89)
                .append(name)
                .append(dag)
                .toHashCode();
    }
    /**
     * Creates and returns a deep copy of this Bayesian network. This implementation copies everything in it.
     * Consequently, it is safe to do anything you want to the deep copy.
     *
     * @return a deep copy of this network.
     */
    @Override
    public DiscreteBayesNet clone() {
        return new DiscreteBayesNet(this);
    }

    /** {@inheritDoc} */
    public String toString(int amount) {
        // amount cannot be non-negative
        if(amount <= 0)
            throw new IllegalArgumentException("The amount must be positive");

        // prepares white space for indent
        StringBuffer whiteSpace = new StringBuffer();
        for (int i = 0; i < amount; i++) {
            whiteSpace.append('\t');
        }

        // builds string representation
        StringBuffer stringBuffer = new StringBuffer();

        stringBuffer.append(whiteSpace);
        stringBuffer.append(getName() + " {\n");

        stringBuffer.append(whiteSpace);
        stringBuffer.append("\tnumber of nodes = " + getNumberOfNodes() + ";\n");

        stringBuffer.append(whiteSpace);
        stringBuffer.append("\tnodes = {\n");

        for (DiscreteBeliefNode node : this.getNodes()) {
            stringBuffer.append(node.toString(amount + 2));
        }

        stringBuffer.append(whiteSpace);
        stringBuffer.append("\t};\n");

        stringBuffer.append(whiteSpace);
        stringBuffer.append("\tnumber of edges = " + getNumberOfEdges() + ";\n");

        stringBuffer.append(whiteSpace);
        stringBuffer.append("\tedges = {\n");

        for (Edge edge : this.getEdges()) {
            stringBuffer.append(edge.toString(amount + 2));
        }

        stringBuffer.append(whiteSpace);
        stringBuffer.append("\t};\n");

        stringBuffer.append(whiteSpace);
        stringBuffer.append("};\n");

        return stringBuffer.toString();
    }

    /*****************************************************************/
    /***************** MODEL MANIPULATION METHODS ********************/
    /*****************************************************************/

    /**
     * The argument variable's cardinality (number of cardinal states) is increased by the the argument amount. This method creates a new BN object
     * with modified parent edges & nodes, and children edges & nodes.
     *
     * @param latentVar the variable whose cardinality is increased.
     * @param amount the amount of cardinality.
     * @return the updated Bayesian network.
     */
    public DiscreteBayesNet increaseCardinality(DiscreteVariable latentVar, int amount) {

        if(!this.containsVar(latentVar))
            throw new IllegalArgumentException("The latent variable doesn't belong to this BayesNet");

        if(!latentVar.isLatentVariable())
            throw new IllegalArgumentException(latentVar.getName()+" is not a latent variable");

        /* 1 - Clonamos el modelo para no modificar la red actual */
        DiscreteBayesNet candModel = this.clone();

        /* 2 - Preparamos el camino para la nueva variable latente */
        DiscreteBeliefNode latentVarNode = candModel.getNode(latentVar);

        // TODO: 30-11-2018 -> Guardamos los arcos de la antigua variable a la que va a suplantar
        List<Variable> toVariables = new ArrayList<>();
        for (AbstractBeliefNode child : latentVarNode.getChildrenNodes())
            toVariables.add(child.getVariable());

        List<Variable> fromVariables = new ArrayList<>();
        for (AbstractBeliefNode parent : latentVarNode.getParentNodes())
            fromVariables.add(parent.getVariable());

        // TODO: 30-11-2018 -> Eliminamos la antigua variable (su nodo) junto a todos sus arcos en el modelo
        candModel.removeNode(latentVarNode);

        /* 3 - Creamos la nueva variable latente con la cardinalidad incrementada y le asignamos los arcos de la antigua */
        // Then we add a new latent var node with the increased cardinality
        DiscreteVariable newLatentVar = new DiscreteVariable(latentVar.getCardinality() + amount, VariableType.LATENT_VARIABLE);
        // Rename the new LV as the old LV
        newLatentVar.setName(latentVar.getName());
        // TODO: 30-11-2018 -> Le asignamos el indice de la antigua para que no la considere una nueva variable
        newLatentVar.setIndex(latentVar.getIndex());

        // TODO: 30-11-2018 -> Añadimos la nueva variable como un nodo y sus arcos asociados
        DiscreteBeliefNode newLatentVarNode = candModel.addNode(newLatentVar);

        // TODO: 07-02-2019
        /* 4 - Añadismos los antiguos arcos de la variable latente eliminada */
        for(Variable toVar: toVariables)
            candModel.addEdge(candModel.getNode(toVar), newLatentVarNode);

        for(Variable fromVar: fromVariables)
            candModel.addEdge(newLatentVarNode, candModel.getNode(fromVar));

        /* 5 - Dado que hemos cambiado la estructura, le asignamos parametros aleatorios para forzar un reaprendizaje completo */
        // TODO: 30-11-2018 -> Aqui se podria hacer que se mantuviesen los parametros de iteraciones previas para temas del EM
        candModel.randomlyParameterize();
        return candModel;
    }

    /**
     * The argument variable's cardinality (number of cardinal states) is increased by the the argument amount. This method creates a new BN object
     * with modified parent edges & nodes, and children edges & nodes.
     *
     * @param latentVar the variable whose cardinality is increased.
     * @param amount the amount of cardinality.
     * @return the updated Bayesian network.
     */
    public DiscreteBayesNet decreaseCardinality(DiscreteVariable latentVar, int amount) {

        if(!this.containsVar(latentVar))
            throw new IllegalArgumentException("The latent variable doesn't belong to this BayesNet");

        if(!latentVar.isLatentVariable())
            throw new IllegalArgumentException(latentVar.getName()+" is not a latent variable");

        if((latentVar.getCardinality() - amount) < 2)
            throw new IllegalArgumentException("The resulting cardinality cannot be lower than 2");

        /* 1 - Clonamos el modelo para no modificar la red actual */
        DiscreteBayesNet candModel = this.clone();

        /* 2 - Preparamos el camino para la nueva variable latente */
        DiscreteBeliefNode latentVarNode = candModel.getNode(latentVar);

        // TODO: 30-11-2018 -> Guardamos los arcos de la antigua variable a la que va a suplantar
        List<Variable> toVariables = new ArrayList<>();
        for (AbstractBeliefNode child : latentVarNode.getChildrenNodes())
            toVariables.add(child.getVariable());

        List<Variable> fromVariables = new ArrayList<>();
        for (AbstractBeliefNode parent : latentVarNode.getParentNodes())
            fromVariables.add(parent.getVariable());

        // TODO: 30-11-2018 -> Eliminamos la antigua variable (su nodo) junto a todos sus arcos en el modelo
        candModel.removeNode(latentVarNode);

        /* 3 - Creamos la nueva variable latente con la cardinalidad incrementada y le asignamos los arcos de la antigua */
        // Then we add a new latent var node with the increased cardinality
        DiscreteVariable newLatentVar = new DiscreteVariable(latentVar.getCardinality() - amount, VariableType.LATENT_VARIABLE);
        // Rename the new LV as the old LV
        newLatentVar.setName(latentVar.getName());
        // TODO: 30-11-2018 -> Le asignamos el indice de la antigua para que no la considere una nueva variable
        newLatentVar.setIndex(latentVar.getIndex());

        // TODO: 30-11-2018 -> Añadimos la nueva variable como un nodo y sus arcos asociados
        DiscreteBeliefNode newLatentVarNode = candModel.addNode(newLatentVar);

        // TODO: 07-02-2019
        /* 4 - Añadismos los antiguos arcos de la variable latente eliminada */
        for(Variable toVar: toVariables)
            candModel.addEdge(candModel.getNode(toVar), newLatentVarNode);

        for(Variable fromVar: fromVariables)
            candModel.addEdge(newLatentVarNode, candModel.getNode(fromVar));

        /* 4 - Dado que hemos cambiado la estructura, le asignamos parametros aleatorios para forzar un reaprendizaje completo */
        // TODO: 30-11-2018 -> Aqui se podria hacer que se mantuviesen los parametros de iteraciones previas para temas del EM
        candModel.randomlyParameterize();
        return candModel;
    }

    /**
     * The argument variable's cardinality (number of cardinal states) is increased by the the argument amount. This method creates a new BN object
     * with modified parent edges & nodes, and children edges & nodes.
     *
     * @param latentVar the variable whose cardinality is increased.
     * @param amount the amount of cardinality.
     * @return the updated Bayesian network.
     */
    public DiscreteBayesNet increaseCardinality(DiscreteVariable latentVar, int amount, Random random) {

        if(!this.containsVar(latentVar))
            throw new IllegalArgumentException("The latent variable doesn't belong to this BayesNet");

        if(!latentVar.isLatentVariable())
            throw new IllegalArgumentException(latentVar.getName()+" is not a latent variable");

        /* 1 - Clonamos el modelo para no modificar la red actual */
        DiscreteBayesNet candModel = this.clone();

        /* 2 - Preparamos el camino para la nueva variable latente */
        DiscreteBeliefNode latentVarNode = candModel.getNode(latentVar);

        // TODO: 30-11-2018 -> Guardamos los arcos de la antigua variable a la que va a suplantar
        List<Variable> toVariables = new ArrayList<>();
        for (AbstractBeliefNode child : latentVarNode.getChildrenNodes())
            toVariables.add(child.getVariable());

        List<Variable> fromVariables = new ArrayList<>();
        for (AbstractBeliefNode parent : latentVarNode.getParentNodes())
            fromVariables.add(parent.getVariable());

        // TODO: 30-11-2018 -> Eliminamos la antigua variable (su nodo) junto a todos sus arcos en el modelo
        candModel.removeNode(latentVarNode);

        /* 3 - Creamos la nueva variable latente con la cardinalidad incrementada y le asignamos los arcos de la antigua */
        // Then we add a new latent var node with the increased cardinality
        DiscreteVariable newLatentVar = new DiscreteVariable(latentVar.getCardinality() + amount, VariableType.LATENT_VARIABLE);
        // Rename the new LV as the old LV
        newLatentVar.setName(latentVar.getName());
        // TODO: 30-11-2018 -> Le asignamos el indice de la antigua para que no la considere una nueva variable
        newLatentVar.setIndex(latentVar.getIndex());

        // TODO: 30-11-2018 -> Añadimos la nueva variable como un nodo y sus arcos asociados
        DiscreteBeliefNode newLatentVarNode = candModel.addNode(newLatentVar);

        // TODO: 07-02-2019
        /* 4 - Añadismos los antiguos arcos de la variable latente eliminada */
        for(Variable toVar: toVariables)
            candModel.addEdge(candModel.getNode(toVar), newLatentVarNode);

        for(Variable fromVar: fromVariables)
            candModel.addEdge(newLatentVarNode, candModel.getNode(fromVar));

        /* 5 - Dado que hemos cambiado la estructura, le asignamos parametros aleatorios para forzar un reaprendizaje completo */
        // TODO: 30-11-2018 -> Aqui se podria hacer que se mantuviesen los parametros de iteraciones previas para temas del EM
        candModel.randomlyParameterize();
        return candModel;
    }

    /**
     * The argument variable's cardinality (number of cardinal states) is increased by the the argument amount. This method creates a new BN object
     * with modified parent edges & nodes, and children edges & nodes.
     *
     * @param latentVar the variable whose cardinality is increased.
     * @param amount the amount of cardinality.
     * @return the updated Bayesian network.
     */
    public DiscreteBayesNet decreaseCardinality(DiscreteVariable latentVar, int amount, Random random) {

        if(!this.containsVar(latentVar))
            throw new IllegalArgumentException("The latent variable doesn't belong to this BayesNet");

        if(!latentVar.isLatentVariable())
            throw new IllegalArgumentException(latentVar.getName()+" is not a latent variable");

        if((latentVar.getCardinality() - amount) < 2)
            throw new IllegalArgumentException("The resulting cardinality cannot be lower than 2");

        /* 1 - Clonamos el modelo para no modificar la red actual */
        DiscreteBayesNet candModel = this.clone();

        /* 2 - Preparamos el camino para la nueva variable latente */
        DiscreteBeliefNode latentVarNode = candModel.getNode(latentVar);

        // TODO: 30-11-2018 -> Guardamos los arcos de la antigua variable a la que va a suplantar
        List<Variable> toVariables = new ArrayList<>();
        for (AbstractBeliefNode child : latentVarNode.getChildrenNodes())
            toVariables.add(child.getVariable());

        List<Variable> fromVariables = new ArrayList<>();
        for (AbstractBeliefNode parent : latentVarNode.getParentNodes())
            fromVariables.add(parent.getVariable());

        // TODO: 30-11-2018 -> Eliminamos la antigua variable (su nodo) junto a todos sus arcos en el modelo
        candModel.removeNode(latentVarNode);

        /* 3 - Creamos la nueva variable latente con la cardinalidad incrementada y le asignamos los arcos de la antigua */
        // Then we add a new latent var node with the increased cardinality
        DiscreteVariable newLatentVar = new DiscreteVariable(latentVar.getCardinality() - amount, VariableType.LATENT_VARIABLE);
        // Rename the new LV as the old LV
        newLatentVar.setName(latentVar.getName());
        // TODO: 30-11-2018 -> Le asignamos el indice de la antigua para que no la considere una nueva variable
        newLatentVar.setIndex(latentVar.getIndex());

        // TODO: 30-11-2018 -> Añadimos la nueva variable como un nodo y sus arcos asociados
        DiscreteBeliefNode newLatentVarNode = candModel.addNode(newLatentVar);

        // TODO: 07-02-2019
        /* 4 - Añadismos los antiguos arcos de la variable latente eliminada */
        for(Variable toVar: toVariables)
            candModel.addEdge(candModel.getNode(toVar), newLatentVarNode);

        for(Variable fromVar: fromVariables)
            candModel.addEdge(newLatentVarNode, candModel.getNode(fromVar));

        /* 4 - Dado que hemos cambiado la estructura, le asignamos parametros aleatorios para forzar un reaprendizaje completo */
        // TODO: 30-11-2018 -> Aqui se podria hacer que se mantuviesen los parametros de iteraciones previas para temas del EM
        candModel.randomlyParameterize();
        return candModel;
    }


    /**
     * TODO: 29/11/2018 Metodo especifico para ajustar los indices cuando no coinciden y queremos aprender el modelo partiendo de este
     */
    public void adjustVariableIndexes(List<DiscreteVariable> variablesWithCorrectIndexes) {
        for(DiscreteVariable variableWithCorrectIndex: variablesWithCorrectIndexes)
            for(DiscreteVariable variable: this.getVariables())
                if(variableWithCorrectIndex.equals(variable))
                    variable.setIndex(variableWithCorrectIndex.getIndex());
    }
}