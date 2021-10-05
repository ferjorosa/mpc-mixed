package experiments.util;

import eu.amidst.core.datastream.Attribute;
import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.datastream.DataOnMemoryListContainer;
import eu.amidst.core.datastream.filereaders.arffFileReader.ARFFDataWriter;
import eu.amidst.core.io.DataStreamLoader;
import eu.amidst.extension.data.DataUtils;
import eu.amidst.extension.util.MyMath;
import eu.amidst.extension.util.tuple.Tuple2;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

public class GenerateLatentData {

    public static void main(String[] args) throws Exception {

        int k = 10; // 10 folds per file

        String[] dataTypes = {"discrete", "continuous", "mixed"};

        String inputBasePath = "data";
        String outputBasePath = "latent_data";

        for(String dataType: dataTypes) {

            String inputDirectory = inputBasePath + "/" + dataType;
            String outputDirectory = outputBasePath + "/" + dataType;

            File f_directory = new File(inputDirectory);
            String[] fileNames = f_directory.list(new FilenameFilter() {
                @Override
                public boolean accept(File f, String name) {
                    return name.endsWith(".arff");
                }
            });

            for (String fileName : fileNames) {
                String dataName = fileName.split("\\.")[0];
                System.out.println("\n" + dataName);
                String filePath = inputDirectory + "/" + fileName;

                DataOnMemory<DataInstance> data = DataStreamLoader.open(filePath).toDataOnMemory();
                List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds = generate(data, k);
                export(folds, dataName, outputDirectory + "/" + dataName + "/" + k + "_folds/");
            }
        }
    }

    public static List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> generateAndExport(DataOnMemory<DataInstance> data,
                                                                                                         int k,
                                                                                                         String dataName,
                                                                                                         String path) throws IOException {

        List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> trainTestFolds = generate(data, k);

        for(int i = 0; i < trainTestFolds.size(); i++){
            Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>> fold = trainTestFolds.get(i);
            ARFFDataWriter.writeToARFFFile(fold.getFirst(), path + dataName + "_" + (i+1) + "_train.arff");
            ARFFDataWriter.writeToARFFFile(fold.getSecond(), path + dataName + "_" + (i+1) + "_test.arff");
        }

        return trainTestFolds;
    }

    public static void export(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                               String dataName,
                               String path) throws IOException {

        /* Create directory if it doesnt exist */
        new File(path).mkdirs();

        for(int i = 0; i < folds.size(); i++){
            Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>> fold = folds.get(i);
            ARFFDataWriter.writeToARFFFile(fold.getFirst(), path + dataName + "_" + (i+1) + "_train.arff");
            ARFFDataWriter.writeToARFFFile(fold.getSecond(), path + dataName + "_" + (i+1) + "_test.arff");
        }
    }


    /** Generates a list of k train-test folds */
    public static List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> generate(DataOnMemory<DataInstance> data, int k) {

        /* Initialize the list of train-set folds */
        List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> trainTestFolds = new ArrayList<>(k);

        /* First, divide the dataset into k folds */
        int[] indices = new int[k+1];
        int division = data.getNumberOfDataInstances() / k;
        for(int i = 1; i < indices.length -1; i++) {
            int t = indices[i-1];
            indices[i] = t + division;
        }
        indices[k] = data.getNumberOfDataInstances();

        List<List<DataInstance>> folds = new ArrayList<>(k);

        for(int i = 0; i < k; i++){
            List<DataInstance> fold = new ArrayList<>(indices[i+1] - indices[i]);
            folds.add(fold);
            for(int j = indices[i]; j < indices[i+1]; j++){
                fold.add(data.getDataInstance(j));
            }
        }

        /* Then rotate these folds to generate a pair of train and test datasets */
        for(int i = 0; i < k; i++) {
            List<DataInstance> trainInstances = new ArrayList<>();
            List<DataInstance> testInstances = new ArrayList<>();
            for(int j = 0; j < k; j++) {
                if (j == i)
                    testInstances.addAll(folds.get(j));
                else
                    trainInstances.addAll(folds.get(j));
            }
            DataOnMemory<DataInstance> foldTrainData = new DataOnMemoryListContainer<>(data.getAttributes(), trainInstances);
            DataOnMemory<DataInstance> foldTestData = new DataOnMemoryListContainer<>(data.getAttributes(), testInstances);
            trainTestFolds.add(new Tuple2<>(foldTrainData, foldTestData));
        }

        /*  IMPORTANT: to filter those columns whose values are constant */
        trainTestFolds = filterZeroVarianceColumns(trainTestFolds);

        return trainTestFolds;
    }

    /** Iterate through the folds, if there is a column in any of them (on the TRAIN part) with zero variance, it is filtered */
    public static List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> filterZeroVarianceColumns(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds) {

        /* Find columns with zero variance*/
        Set<Integer> attributesToFilter = new LinkedHashSet<>();
        for(Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>> fold: folds) {

            DataOnMemory<DataInstance> train = fold.getFirst();

            // Initialize columnsData
            List<double[]> columnsData = new ArrayList<>(train.getAttributes().getNumberOfAttributes());
            for(int i = 0; i < train.getAttributes().getNumberOfAttributes(); i++)
                columnsData.add(new double[train.getNumberOfDataInstances()]);

            // Iterate through the instances and add its data to its corresponding array
            for(int i = 0; i < train.getNumberOfDataInstances(); i++){
                double[] instanceValues = train.getDataInstance(i).toArray();
                for(int j = 0; j < instanceValues.length; j++){
                    columnsData.get(j)[i] = instanceValues[j]; // j columns, i instance
                }
            }

            // Once the data has been separated by column, estimate the variance
            for(int j = 0; j < columnsData.size(); j++) {
                double stdev = MyMath.stDev(columnsData.get(j));
                double var = Math.pow(stdev, 2);
                if(var == 0)
                    attributesToFilter.add(j);
            }
        }

        /* Print the attributes with zero variance that are going to be filtered */
        System.out.println("Attributes with zero variance in at least one fold (will be ignored): ");
        for(Integer zeroVarAttributeIndex: attributesToFilter) {
            Attribute zeroVarAttribute = folds.get(0).getFirst()
                    .getAttributes().getFullListOfAttributes().get(zeroVarAttributeIndex);
            System.out.print(zeroVarAttribute.getName()+", ");
        }
        System.out.println();

        /* Select columns with more than zero variance and filter the others */
        List<Attribute> attributesForProjection = new ArrayList<>();
        List<Attribute> columns = folds.get(0).getFirst().getAttributes().getFullListOfAttributes();
        for(int j = 0; j < columns.size(); j++) {
            if(!attributesToFilter.contains(j))
                attributesForProjection.add(columns.get(j));
        }

        /* Project each fold */
        List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> projectedFolds = new ArrayList<>(folds.size());
        for(Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>> fold: folds) {
            DataOnMemory<DataInstance> train = fold.getFirst();
            DataOnMemory<DataInstance> test = fold.getSecond();
            DataOnMemory<DataInstance> projectedTrain = DataUtils.project(train, attributesForProjection);
            DataOnMemory<DataInstance> projectedTest = DataUtils.project(test, attributesForProjection);
            projectedFolds.add(new Tuple2<>(projectedTrain, projectedTest));
        }

        return projectedFolds;
    }
}
