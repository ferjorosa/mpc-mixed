package methods;

import eu.amidst.core.datastream.DataInstance;
import eu.amidst.core.datastream.DataOnMemory;
import eu.amidst.core.models.BayesianNetwork;
import eu.amidst.extension.io.GenieWriter;
import eu.amidst.extension.util.LogUtils;
import eu.amidst.extension.util.tuple.Tuple2;

import java.io.File;
import java.util.List;

public interface LatentMixedMethod extends LatentMethod {

    void runLatentMixed(List<Tuple2<DataOnMemory<DataInstance>, DataOnMemory<DataInstance>>> folds,
                        String dataName,
                        int run,
                        LogUtils.LogLevel foldLogLevel) throws Exception;

    default void storeMixedModels(List<BayesianNetwork> models,
                                  String directoryPath,
                                  String dataName,
                                  String methodName) throws Exception {

        new File(directoryPath).mkdirs();

        for(int i = 0; i < models.size(); i++) {
            String output = directoryPath + "/" + dataName + "_" + (i+1) + "_" + methodName + ".xdsl";
            GenieWriter genieWriter = new GenieWriter();
            genieWriter.write(models.get(i), new File(output));
        }
    }
}
