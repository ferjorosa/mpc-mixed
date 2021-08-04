package methods;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.tuple.Tuple2;
import voltric.io.model.bif.BnLearnBifFileWriter;
import voltric.model.DiscreteBayesNet;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;

public interface LatentDiscreteMethod extends LatentMethod {

    void runLatentDiscrete(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                           String dataName,
                           int run,
                           LogUtils.LogLevel foldLogLevel) throws Exception;

    default void storeLatentDiscreteModels(List<DiscreteBayesNet> models,
                                           String directoryPath,
                                           String dataName,
                                           String methodName) throws IOException {

        new File(directoryPath).mkdirs();

        for(int i = 0; i < models.size(); i++) {
            String output = directoryPath + "/" + dataName + "_" + (i+1) + "_" + methodName + ".bif";
            BnLearnBifFileWriter writer = new BnLearnBifFileWriter(new FileOutputStream(output));
            writer.write(models.get(i));
        }
    }
}
