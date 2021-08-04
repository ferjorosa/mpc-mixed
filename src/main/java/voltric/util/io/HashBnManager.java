package voltric.util.io;

import voltric.clustering.unidimensional.OldLearnLatentKDB;
import voltric.data.DiscreteData;
import voltric.io.model.bif.OldBifFileReader;
import voltric.io.model.bif.OldBifFileWriter;
import voltric.learning.LearningResult;
import voltric.learning.parameter.DiscreteParameterLearning;
import voltric.learning.parameter.em.ParallelEM;
import voltric.learning.parameter.em.config.EmConfig;
import voltric.learning.parameter.mle.MLE;
import voltric.learning.score.LearningScore;
import voltric.learning.score.ScoreType;
import voltric.learning.structure.constraintbased.HybridPC;
import voltric.model.DiscreteBayesNet;
import voltric.variables.DiscreteVariable;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Collection;
import java.util.List;

/**
 * Solo utilizan las manifestVariable sporque como el hash se hace con el sumatorio de los hashcodes de los strings,
 * es posible que una LV tuviera un nombre no predecible como "variable356"
 */
public class HashBnManager {

    public static DiscreteBayesNet load(String pathFolder, Collection<DiscreteVariable> manifestVariables) throws IOException{
        long hash = SimpleHashCreator.createHash(manifestVariables);

        return OldBifFileReader.readOBif(pathFolder + hash+".bif");
    }

    public static void save(String pathFolder, DiscreteBayesNet bayesNet) throws IOException{
        long hash = SimpleHashCreator.createHash(bayesNet.getManifestVariables());

        OldBifFileWriter.writeBif(pathFolder + hash+".bif", bayesNet);
    }

    //Ademas de cargar la red, la guarda si no la habia aprendido antes
    public static LearningResult<DiscreteBayesNet> loadOrLearnPartition(DiscreteData data,
                                                                        String destinationFolder,
                                                                        double kdBThreshold,
                                                                        List<DiscreteVariable> manifestVariables,
                                                                        HybridPC pc,
                                                                        long seed){

        DiscreteData projectedData = data.project(manifestVariables);
        DiscreteBayesNet loadedBN;
        try{
            try {

                loadedBN = HashBnManager.load(destinationFolder + "partitions/", manifestVariables);

            }catch (FileNotFoundException fileNotFound){

                // Aprendemos or cargamos la red correspondiente a las MVs de la particion
                DiscreteBayesNet mvBnet = loadOrLearnMidNetwork(data, destinationFolder, manifestVariables, pc);
                // Aprendemos la particion correspondiente con su LV
                DiscreteParameterLearning em = new ParallelEM(new EmConfig(seed));
                loadedBN = OldLearnLatentKDB.learnModel(mvBnet, 10, data, em, kdBThreshold, 3, 200).getBayesianNetwork();
                // Guardamos dicha BN
                HashBnManager.save(destinationFolder + "partitions/", loadedBN);
            }
        }catch (IOException ioex){
            throw new UncheckedIOException(ioex);
        }

        double score = LearningScore.calculateBIC(projectedData, loadedBN);
        return new LearningResult<>(loadedBN, score, ScoreType.BIC);
    }

    //
    public static DiscreteBayesNet loadOrLearnMidNetwork(DiscreteData data,
                                                         String destinationFolder,
                                                         List<DiscreteVariable> manifestVariables,
                                                         HybridPC pc){

        DiscreteBayesNet loadedBN;
        DiscreteData projectedData = data.project(manifestVariables);

        try{
            try {
                loadedBN = HashBnManager.load(destinationFolder + "midNets/", manifestVariables);
                return loadedBN;
            }catch (FileNotFoundException fileNotFound){
                // Aprendemos la red correspondiente a las MVs de la particion
                loadedBN = pc.learnModel(projectedData, new MLE(ScoreType.BIC)).getBayesianNetwork();
                // Guardamos dicha BN
                HashBnManager.save(destinationFolder + "midNets/", loadedBN);
            }
        } catch (IOException ioex){
            throw new UncheckedIOException(ioex);
        }
        return loadedBN;
    }
}
