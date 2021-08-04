package voltric.util.weka.data;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import java.io.File;

/**
 * Created by equipo on 17/11/2017.
 */
public class CSV2Arff {

    public static void main(String[] args) throws Exception {

        // load CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("data/continuous/low_medium/ecoli/ecoli_2.csv"));
        Instances data = loader.getDataSet();

        // save ARFF
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File("data/continuous/low_medium/ecoli/ecoli_2.arff"));
        saver.writeBatch();
    }
}
