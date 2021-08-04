package voltric.io.clustering;

import voltric.data.DiscreteData;
import voltric.data.DiscreteDataInstance;
import voltric.util.Tuple;
import voltric.variables.DiscreteVariable;
import voltric.variables.IVariable;
import voltric.variables.StateSpaceType;

import java.io.FileWriter;
import java.io.IOException;
import java.util.List;

/**
 * Created by equipo on 05/02/2018.
 */
public class ExportUnidimensionalClusterAssignments {

    // Genera un nuevo archivo ARFF a partir del cual se produjeron los assignments
    public static void export(DiscreteData data, List<Tuple<DiscreteDataInstance, double[]>> assignments, String filePathString) throws IOException {
        FileWriter fw = new FileWriter(filePathString);

        int clusterCardinality = assignments.get(0).getSecond().length;

        // Writes the ARFF @relation line that identifies the Data
        fw.write("@relation cluster_assigned_card_" + clusterCardinality +"_" + data.getName() + "\n\n");

        // Writes the ARFF attributes
        for (DiscreteVariable att : data.getVariables()) {
            fw.write(attributeToArffString(att) + "\n");
        }

        // Writes the ARFF cluster probab as attributes
        for(int i = 0; i < clusterCardinality; i++) {
            fw.write("@attribute c_" + i + " numeric" + "\n");
        }

        // Writes the ARFF data instances (where eac cluster label has a probability value)
        fw.write("\n@data\n");

        for (Tuple<DiscreteDataInstance, double[]> assignment: assignments)
            writeInstanceWithclustersToFile(assignment.getFirst(), assignment.getSecond(), fw, ",");

        // Closes the file
        fw.close();
    }

    private static void writeInstanceWithclustersToFile(DiscreteDataInstance instance,
                                                        double[] clusterProbabilities,
                                                        FileWriter writer,
                                                        String separator) throws IOException{
        // instance string
        String instanceString = instanceToArffString(instance, separator);

        instanceString = attachClusterProbabilities(clusterProbabilities, instanceString, separator);

        int weight =  instance.getData().getWeight(instance);

        for(int i=0; i < weight; i++)
            writer.write(instanceString + "\n");
    }

    private static String instanceToArffString(DiscreteDataInstance instance, String separator) {
        String s = "";

        // Append all the columns of the DataInstance with  the separator except the last one
        for(int i = 0; i < instance.getTextualValues().size() - 1; i++)
            s += instance.getTextualValue(i) + separator;
        // Append the last column of the instance without the separator
        s += instance.getTextualValue(instance.getTextualValues().size() - 1);
        return s;
    }

    private static String attachClusterProbabilities(double[] clusterProbabilities, String instanceString, String separator) {
        // Attach cluster label probability
        for(int i = 0; i< clusterProbabilities.length; i++)
            instanceString += separator + clusterProbabilities[i];

        return instanceString;
    }

    /**
     * Transforms a variable into an equivalent ARFF @attribute line.
     *
     * @param attribute the variable that is going to be transformed.
     * @param <V> the specific type of the variable ({@code DiscreteVariable}, {@code AbstractContinuousVariable}, {@code Variable}, etc.)
     * @return the equivalent @attribute line.
     */
    private static <V extends IVariable> String attributeToArffString(V attribute){
        if(attribute.getStateSpaceType() == StateSpaceType.REAL)
            return "@attribute " + attribute.getName() + " real";
        else if(attribute.getStateSpaceType() == StateSpaceType.FINITE) {
            StringBuilder stringBuilder = new StringBuilder("@attribute " + attribute.getName() + " {");
            DiscreteVariable discreteAttribute = (DiscreteVariable) attribute;
            List<String> attributeStates = discreteAttribute.getStates();

            // Append all the variable states minus the last one
            attributeStates
                    .stream()
                    .limit(discreteAttribute.getStates().size() - 1)
                    .forEach(e -> stringBuilder.append(e + ", "));

            // Append the last state
            stringBuilder.append(attributeStates.get(attributeStates.size() - 1) + "}");

            return stringBuilder.toString();
        }
        else
            throw new IllegalArgumentException("Unknown SateSapaceType");
    }

    /**
     * Transforms the {@link DiscreteDataInstance} into an equivalent ARFF @data line.
     *
     * @param dataInstance the instance to be transformed into an ARFF string.
     * @return the ARFF @data line equivalent of the dataInstance.
     */
    private static String discreteDataInstanceToARFFString(List<DiscreteVariable> attributes, DiscreteDataInstance dataInstance, String separator){
        StringBuilder builder = new StringBuilder();

        // Append all the columns of the DataInstance with  the separator except the last one
        for(int i=0; i<dataInstance.getTextualValues().size();i++)
            builder.append(discreteDataInstanceToARFFString(attributes.get(i), dataInstance, attributes, ","));

        // Append the last column of the data instance
        DiscreteVariable att = attributes.get(attributes.size() - 1);
        builder.append(discreteDataInstanceToARFFString(att,dataInstance, attributes, ""));

        return builder.toString();
    }

    /**
     * This method returns the string equivalent of an specific value of the instance
     *
     * @param att the variable that indicates the values that is going to be transformed.
     * @param dataInstance the instance being converted into an ARFF string.
     * @param separator the separator being used
     * @return the string equivalent of the numeric value.
     */
    // TODO: Maybe is better not to modularize this part and add it to the "dataInstanceToARFFString" method? (different strategy)
    private static String discreteDataInstanceToARFFString(DiscreteVariable att, DiscreteDataInstance dataInstance, List<DiscreteVariable> variables, String separator) {
        StringBuilder builder = new StringBuilder();
        if(dataInstance.getNumericValue(variables.indexOf(att)) == Double.NaN) // Value is MISSING
            builder.append("?" + separator);
        else if (att.getStateSpaceType() == StateSpaceType.FINITE || att.getStateSpaceType() == StateSpaceType.REAL){
            builder.append(dataInstance.getTextualValue(variables.indexOf(att)) + separator);
        } else
            throw new IllegalArgumentException("Illegal state space type of Attribute: " + att.getStateSpaceType());

        return builder.toString();
    }
}
