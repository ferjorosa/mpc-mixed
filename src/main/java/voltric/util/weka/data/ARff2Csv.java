package voltric.util.weka.data;

import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVSaver;

import java.io.File;

/**
 * Created by equipo on 11/12/2017.
 */
public class ARff2Csv {

    public static void main(String[] args) throws Exception {

        // load Arff
        ArffLoader loader = new ArffLoader();
        loader.setSource(new File("articulo/data/synthetic/10nodes/random10NodesMPC_5000.arff"));
        Instances data = loader.getDataSet();

        // save CSV
        CSVSaver saver = new CSVSaver();
        saver.setInstances(data);
        saver.setFile(new File("articulo/data/synthetic/10nodes/random10NodesMPC_5000.csv"));
        //saver.setDestination(new File("data/parkinson/s6/d897_female_motor_nms55_s6.arff"));
        saver.writeBatch();
    }
}
