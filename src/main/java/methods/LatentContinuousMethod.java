package methods;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.tuple.Tuple2;
import org.latlab.core.io.bif.BifWriter;
import org.latlab.core.model.Gltm;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;

public interface LatentContinuousMethod extends LatentMethod {

    void runLatentContinuous(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                             String dataName,
                             int run,
                             LogUtils.LogLevel foldLogLevel) throws Exception;

    default void storeContinuousModels(List<Gltm> models,
                                       String directoryPath,
                                       String dataName,
                                       String methodName) throws IOException {

        new File(directoryPath).mkdirs();

        for(int i = 0; i < models.size(); i++) {
            String output = directoryPath + "/" + dataName + "_" + (i+1) + "_" + methodName + ".bif";
            BifWriter writer = new BifWriter(new FileOutputStream(output));
            writer.write(models.get(i));
        }
    }

    default void storeContinuousModel(Gltm model,
                                      int fold_index,
                                      String directoryPath,
                                      String dataName,
                                      String methodName) throws IOException {

        new File(directoryPath).mkdirs();
        String output = directoryPath + "/" + dataName + "_" + (fold_index+1) + "_" + methodName + ".bif";
        BifWriter writer = new BifWriter(new FileOutputStream(output));
        writer.write(model);
    }
}
