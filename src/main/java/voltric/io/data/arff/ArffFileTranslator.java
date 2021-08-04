package voltric.io.data.arff;

import voltric.data.DiscreteData;
import voltric.data.DiscreteDataInstance;
import voltric.variables.DiscreteVariable;
import voltric.variables.IVariable;
import voltric.variables.StateSpaceType;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Traducimos los nombres de los estados de las variables (i.e "[0.12]") por sus indices, de tal manera que no interfiera
 * con el BIF 0.15 Parser.
 */
public class ArffFileTranslator {

    public static void translateToFile(DiscreteData data, String filePathString) throws IOException {
        FileWriter fw = new FileWriter(filePathString);

        // Writes the ARFF @relation line that identifies the Data
        fw.write("@relation " + data.getName() + "\n\n");

        // Writes the ARFF attributes
        for (DiscreteVariable att : data.getVariables()) {
            fw.write(attributeToArffString(att) + "\n");
        }

        // Writes the ARFF data instances
        fw.write("\n@data\n");

        for (DiscreteDataInstance instance : data.getInstances())
            writeInstanceToFile(instance, fw, ",");

        // Closes the file
        fw.close();
    }

    /** Aqui cambiamos el nombre de los atributos por su indice */
    private static <V extends IVariable> String attributeToArffString(V attribute){
        if(attribute.getStateSpaceType() == StateSpaceType.REAL)
            return "@attribute " + attribute.getName() + " real";
        else if(attribute.getStateSpaceType() == StateSpaceType.FINITE) {
            StringBuilder stringBuilder = new StringBuilder("@attribute " + attribute.getName() + " {");
            DiscreteVariable discreteAttribute = (DiscreteVariable) attribute;

            /** Aqui es donde se hace la sustitucion */
            List<String> translatedAttributeStates = new ArrayList<>();
            for(int i=0; i< discreteAttribute.getCardinality(); i++)
               translatedAttributeStates.add(""+ i);

            List<String> attributeStates = translatedAttributeStates;

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
            throw new IllegalArgumentException("Unknown SateSpaceType");
    }

    private static void writeInstanceToFile(DiscreteDataInstance instance, FileWriter writer, String separator) throws IOException{
        String instanceString = instanceToArffString(instance, separator);
        int weight =  instance.getData().getWeight(instance);

        for(int i=0; i < weight; i++)
            writer.write(instanceString + "\n");
    }

    /** Aqui tambien cambiamos el nombre de los atributos por su indice */
    private static String instanceToArffString(DiscreteDataInstance instance, String separator) {
        String s = "";

        // Append all the columns of the DataInstance with  the separator except the last one
        for(int i = 0; i < instance.getTextualValues().size() - 1; i++)
            s += instance.getNumericValue(i) + separator;
        // Append the last column of the instance without the separator
        s += instance.getNumericValue(instance.getTextualValues().size() - 1);
        return s;
    }
}
