package voltric.data;

import voltric.util.Tuple;
import voltric.variables.DiscreteVariable;

import java.util.*;
import java.util.stream.Collectors;

/**
 * TODO: No deberia permitir variables repetidas, quizas se peude resolver mediante un check en el constructor con contains, o con un Set interno
 * en vez de aceptar la referencia de variables (CACA)
 */
public class DiscreteData {

    public static int MISSING_VALUE = -1;

    /** The associated name. */
    private String name;

    /** The data rows. */
    private List<DiscreteDataInstance> instances = new ArrayList<>();

    /** The data columns. */
    private List<DiscreteVariable> variables = new ArrayList<>();

    /**
     * The weight associated to each data instance.
     */
    private Map<DiscreteDataInstance, Integer> instanceWeights = new HashMap<>();

    private int totalWeight;

    /**
     * Constructs a new {@code Data} object by providing its collection of instances, variables and its name.
     *
     * @param name the name of the data.
     * @param variables the data columns.
     */
    public DiscreteData(String name, List<DiscreteVariable> variables) {
        this.name = name;
        this.variables = new ArrayList<>(variables);
        this.totalWeight = 0;
    }

    /**
     * Constructs a new {@code Data} object with a name by default.
     *
     * @param variables the data columns.
     */
    public DiscreteData(List<DiscreteVariable> variables) {
        this.name = "data";
        this.variables = new ArrayList<>(variables);
        this.totalWeight = 0;
    }

    /**
     * Returns the name of the {@code Data} object.
     *
     * @return the name of the {@code Data} object.
     */
    public String getName() {
        return name;
    }

    /**
     * Modifies the name of {@code Data} by assigning a new name.
     *
     * @param name Data's new name.
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * @return Returns its collection of variables.
     */
    public List<DiscreteVariable> getVariables() {
        return variables;
    }

    /**
     * Returns a specific set of variables given their names.
     *
     * @param names
     * @return
     */
    public List<DiscreteVariable> getVariables(List<String> names) {
        return this.getVariables().stream().filter(x->names.contains(x.getName())).collect(Collectors.toList());
    }

    /**
     * If exists, returns a specific variable given its name.
     *
     * @param name
     * @return
     */
    public Optional<DiscreteVariable> getVariable(String name) {
        for(DiscreteVariable var: variables)
            if(var.getName().equals(name))
                return Optional.of(var);
        return Optional.empty();
    }

    /**
     * @return Returns its collection of data instances.
     */
    public List<DiscreteDataInstance> getInstances() {
        return instances;
    }

    /**
     * Returns the weight of the instance, its number of repetitions. This method allows an instance whose attributes are
     * a subset of the dataSet variables.
     *
     * @return the weight of the instance, its number of repetitions.
     */
    public int getWeight(DiscreteDataInstance dataInstance){
        return instanceWeights.get(dataInstance);
    }

    /**
     * Modifies the weight of the instance by setting a new value.
     *
     * @param weight the new weight of the instance.
     */
    public void setWeight(DiscreteDataInstance dataInstance, int weight){
        this.instanceWeights.put(dataInstance, weight);
    }

    public int getTotalWeight(){
        return this.totalWeight;
    }

    public boolean hasMissingValues(){
        return false;
    }

    public void add(DiscreteDataInstance dataInstance, int weight){

        dataInstance.setData(this);

        // finds the position for this data instance
        int index = instances.indexOf(dataInstance);

        if(index < 0) {
            // check if the instance is permitted
            if (!this.isInstancePermitted(dataInstance))
                throw new IllegalArgumentException("Data instance is not permitted");

            // adds unseen data case
            instances.add(dataInstance);
            instanceWeights.put(dataInstance, weight);
        }else{
            // increases weight for the existing data instance
            instanceWeights.put(dataInstance, instanceWeights.get(dataInstance) + weight);
        }

        this.totalWeight += weight;
    }

    public void add(DiscreteDataInstance dataInstance){
        this.add(dataInstance, 1);
    }

    /**
     * Projects current data to a new dimension, thus generating a new {@code Data} object.
     *
     * @param variableList the subset of variables that conforms the new dimension of data.
     * @return the projected {@code Data} object.
     */
    // TODO: Añade variables repetidas
    public DiscreteData project(List<DiscreteVariable> variableList){

        if(!this.getVariables().containsAll(variableList))
            throw new IllegalArgumentException("All the argument variables must be involved in this Data object");

        if(this.getVariables().size() == variableList.size() && this.getVariables().containsAll(variableList))
            return this;

        // The projected variable list needs to be ordered according to the original DataSet
        List<DiscreteVariable> orderedList = sortWithOriginalIndex(variableList);

        DiscreteData projectedData = new DiscreteData(orderedList);

        for(DiscreteDataInstance dataInstance: this.instances) {
            int[] projectedValues = new int[orderedList.size()];

            for(DiscreteVariable variable: orderedList)
                projectedValues[orderedList.indexOf(variable)] = dataInstance.getNumericValue(variable);

            DiscreteDataInstance projectedInstance = new DiscreteDataInstance(projectedValues);
            projectedData.add(projectedInstance, this.instanceWeights.get(dataInstance));
        }

        return projectedData;
    }

    /**
     * Projects current data to a new one-dimension, thus generating a new {@code Data} object.
     *
     * @param variable the variable that conforms the new dimension of data.
     * @return the projected {@code Data} object.
     * @see DiscreteData#project(List)
     */
    public DiscreteData project(DiscreteVariable variable){
        List<DiscreteVariable> variableList = new ArrayList<>();
        variableList.add(variable);
        return this.project(variableList);
    }

    // For projection purposes
    private List<DiscreteVariable> sortWithOriginalIndex(List<DiscreteVariable> variables){

        Comparator<Tuple<Integer, DiscreteVariable>> byIndex =
                (pair1, pair2) -> Integer.compare(pair1.getFirst(), pair2.getFirst());

        // Pseudo zip with index
        return this.variables.stream()
                .filter(variables::contains)
                .map(x-> new Tuple<>(this.variables.indexOf(x), x))
                .sorted(byIndex)
                .map(Tuple::getSecond)
                .collect(Collectors.toList());
    }

    public double getFrequency(DiscreteDataInstance instance) {

        if(instance.getVariables().size() < this.getVariables().size())
            throw new IllegalArgumentException("this.varsSize < instance.varsSize. Projected instances are not allowed");
        else if(!instance.getVariables().containsAll(this.getVariables()))
            throw new IllegalArgumentException("instance has to contain all the variables in the dataSet");
        else{
            DiscreteDataInstance projectedDataInstance = instance.project(this);
            return this.getWeight(projectedDataInstance) / this.getTotalWeight();
        }

        /*if(!this.getVariables().containsAll(variables))
            throw new IllegalArgumentException("All the argument variables must be involved in this Data object");

        if(this.getVariables().equals(variables))
            return this.instanceWeights.get(instance) / this.getTotalWeight();

        return instanceWeights.keySet().stream()
                .filter(dataInstance -> dataInstance.getVariables().containsAll(variables))
                .mapToInt(this.instanceWeights::get)
                .sum() / this.getTotalWeight();
        */
    }

    /**
     * Checks if each of the instance's values belong to the state space of its associated variable.
     *
     * @param instance the instance to be checked
     * @return true if all the values belong to the associated state space.
     */
    public boolean isInstancePermitted(DiscreteDataInstance instance){
        if(instance.size() != this.variables.size())
            return false;

        for(int i=0; i < instance.getNumericValues().length; i++)
            if(!this.variables.get(i).isValuePermitted(instance.getNumericValue(i)))
                return false;

        return true;
    }

    /** TODO: Ha sido diseñado para el MyLocalHillClimbing */
    /** It should be ordered */
    public DiscreteData projectV2(List<DiscreteVariable> variableList){

        if(!this.getVariables().containsAll(variableList))
            throw new IllegalArgumentException("All the argument variables must be involved in this Data object");

        if(this.getVariables().size() == variableList.size() && this.getVariables().containsAll(variableList))
            return this;

        // The projected variable list needs to be ordered according to the original DataSet
        List<Tuple<Integer, DiscreteVariable>> orderedListWithIndex = sortAndPairWithOriginalIndex(variableList);
        List<DiscreteVariable> orderedList = orderedListWithIndex.stream().map(pair->pair.getSecond()).collect(Collectors.toList());
        List<Integer> orderedIndexes = orderedListWithIndex.stream().map(pair->pair.getFirst()).collect(Collectors.toList());

        DiscreteData projectedData = new DiscreteData(orderedList);

        for(DiscreteDataInstance dataInstance: this.instances) {
            int[] projectedValues = new int[orderedList.size()];
            for(int i = 0; i < orderedIndexes.size(); i++)
                projectedValues[i] = dataInstance.getNumericValue(orderedIndexes.get(i));

            DiscreteDataInstance projectedInstance = new DiscreteDataInstance(projectedValues);
            projectedData.add(projectedInstance, this.instanceWeights.get(dataInstance));
        }

        return projectedData;
    }

    public List<Tuple<Integer, DiscreteVariable>> sortAndPairWithOriginalIndex(List<DiscreteVariable> variables){

        Comparator<Tuple<Integer, DiscreteVariable>> byIndex =
                (pair1, pair2) -> Integer.compare(pair1.getFirst(), pair2.getFirst());

        // Pseudo zip with index
        return this.variables.stream()
                .filter(variables::contains)
                .map(x-> new Tuple<>(this.variables.indexOf(x), x))
                .sorted(byIndex)
                .collect(Collectors.toCollection(ArrayList::new));
    }

    // TODO: Version que intenta reducir los tiempos de ejecucion
    public DiscreteData projectV3(List<DiscreteVariable> variableList){

        if(!this.getVariables().containsAll(variableList))
            throw new IllegalArgumentException("All the argument variables must be involved in this Data object");

        if(this.getVariables().size() == variableList.size() && this.getVariables().containsAll(variableList))
            return this;

        // The projected variable list needs to be ordered according to the original DataSet
        List<Tuple<Integer, DiscreteVariable>> orderedListWithIndex = sortAndPairWithOriginalIndex(variableList);
        List<DiscreteVariable> orderedList = orderedListWithIndex.stream().map(pair->pair.getSecond()).collect(Collectors.toList());
        List<Integer> orderedIndexes = orderedListWithIndex.stream().map(pair->pair.getFirst()).collect(Collectors.toList());

        DiscreteData projectedData = new DiscreteData(this.name, orderedList);

        for(DiscreteDataInstance dataInstance: this.instances) {
            int[] projectedValues = new int[orderedList.size()];
            for(int i = 0; i < orderedIndexes.size(); i++)
                projectedValues[i] = dataInstance.getNumericValue(orderedIndexes.get(i));

            DiscreteDataInstance projectedInstance = new DiscreteDataInstance(projectedValues);
            projectedData.add(projectedInstance, this.instanceWeights.get(dataInstance));
        }

        return projectedData;
    }
}
