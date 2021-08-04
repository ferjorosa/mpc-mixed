package experiments.util;

import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.datastream.Attributes;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.core.variables.stateSpaceTypes.FiniteStateSpace;
import eu.amidst.extension.util.tuple.Tuple2;

import java.io.File;
import java.io.FileWriter;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/* EAST is a discrete method and this script has been generated with that condition in mind...*/
public class TransformSpanishLivingConditionsDataToEAST {

    public static void main(String[] args) throws Exception {

        int k = 10; // 10 folds per file

        String inputDirectory = "data/spanish_living_conditions/";
        String outputDirectory = "data/spanish_living_conditions/";

        String fileName = "spanish_living_conditions_train.arff";

        String dataName = fileName.split("\\.")[0];
        System.out.println("\n" + dataName);
        String filePath = inputDirectory + "/" + fileName;

        DataOnMemory<DataInstance> data = DataStreamLoader.open(filePath).toDataOnMemory();

        Map<DataInstance, Integer> dataCounts = countInstanceRepetitions(data);
        String outputTrainPath = outputDirectory;
        String outputTrainFile = dataName + ".data";
        generateFile(dataCounts, data.getAttributes(), outputTrainPath, outputTrainFile, dataName);
    }

    private static Map<DataInstance, Integer> countInstanceRepetitions(DataOnMemory<DataInstance> data) {
        Map<DataInstance, Integer> counts = new HashMap<>();
        for (DataInstance instance: data) {
            if(counts.containsKey(instance))
                counts.put(instance, counts.get(instance) + 1);
            else
                counts.put(instance, 1);
        }
        return counts;
    }

    private static void generateFile(Map<DataInstance, Integer> instanceCounts,
                                     Attributes attributes,
                                     String path,
                                     String fileName,
                                     String dataName) throws IOException {

        /* Create the hierarchy of directories (if necessary) */
        new File(path).mkdirs();

        String filePath = path + fileName;
        FileWriter fw = new FileWriter(filePath);

        /* Write data name */
        fw.write("Name: "+ dataName);

        /* Write attributes */
        fw.write("\n\n//Variables: name of variable followed by names of states\n");
        for(Attribute attribute: attributes) {
            fw.write("\n" + attribute.getName()+": ");
            FiniteStateSpace stateSpaceType = attribute.getStateSpaceType();
            for(String state: stateSpaceType)
                fw.write(state+" ");
        }

        /* Write instances */
        fw.write("\n\n//Records: Numbers in the last column are frequencies.\n\n");
        for(DataInstance instance: instanceCounts.keySet())
            writeInstanceToFile(instance, instanceCounts.get(instance), fw, " ", attributes.getFullListOfAttributes());

        fw.close();
    }

    private static void writeInstanceToFile(DataInstance instance,
                                            double instanceWeight, // EAST format requires instance weight to be double-type
                                            FileWriter writer,
                                            String separator,
                                            List<Attribute> attributes) throws IOException{
        try {
            String instanceString = instanceToString(instance, separator, attributes);
            writer.write(instanceString + "   " + instanceWeight + "\n");
        } catch (Exception e) {
            throw e;
        }
    }

    private static String instanceToString(DataInstance instance, String separator, List<Attribute> attributes) {
        String s = "";
        Attribute attribute;
        FiniteStateSpace stateSpace;

        /* Append all the columns of the DataInstance with  the separator except the last one */
        for(int i = 0; i < attributes.size() - 1; i++) {
            attribute = attributes.get(i);
            stateSpace = attribute.getStateSpaceType(); // It should always be discrete
            String nameState = stateSpace.getStatesName((int) instance.getValue(attribute));
            s += nameState + separator;
        }
        /* Append the last column of the instance without the separator */
        attribute = attributes.get(attributes.size() - 1);
        stateSpace = attribute.getStateSpaceType(); // It should always be discrete
        String nameState = stateSpace.getStatesName((int) instance.getValue(attribute));
        s += nameState + separator;

        return s;
    }
}
