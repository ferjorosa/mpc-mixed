package voltric.clustering.util;

import voltric.data.DiscreteData;
import voltric.data.DiscreteDataInstance;
import voltric.model.DiscreteBayesNet;
import voltric.util.Tuple;
import voltric.variables.DiscreteVariable;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by equipo on 12/05/2018.
 *
 * Genera un dataSet completo al asignar a cada instancia su valor de cluster mas probable, con respecto a 1 o mas particiones.
 */
public class GenerateCompleteData {

    public static DiscreteData generateUnidimensional(DiscreteData dataSet, DiscreteBayesNet lcm) {

        if(lcm.getLatentVariables().size() > 1)
            throw new IllegalArgumentException("Solo se permite una variable latente en el modelo");

        Map<DiscreteDataInstance, Integer> completeDataMap = new HashMap<>();
        List<Tuple<DiscreteDataInstance, double[]>> clusterAssingments = AssignToClusters.assignDataCaseToCluster(dataSet, lcm);

        // Inicializamos el Map
        for(DiscreteDataInstance instance: dataSet.getInstances())
            completeDataMap.put(instance, -1);

        // Transformamos las probabilidades de asignacin de cada instancia a un cluster por el indice del cluster con
        // mayor probabilidad y lo almacenamos
        for (Tuple<DiscreteDataInstance, double[]> clustAssignment : clusterAssingments) {

            // Calculamos el índice de valor máximo
            double maxVal = 0;
            int maxIndex = 0;
            for (int i = 0; i < clustAssignment.getSecond().length; i++)
                if (clustAssignment.getSecond()[i] > maxVal) {
                    maxIndex = i;
                    maxVal = clustAssignment.getSecond()[i];
                }

            // Asignamos el indice del cluster al que pertenece la instancia con mayor probabilidad a la instancia en cuestion
            completeDataMap.put(clustAssignment.getFirst(), maxIndex);
        }

        // Una vez hemos asignado a cada instancia su cluster mas probable, generamos un objeto "DiscreteData" donde
        // se incluyan las variables de particion como variables observadas.

        // Por cada variable latente creamos una MV cuyo nombre coincide y que servirá para el nuevo DataSet
        // TODO 09-07-2018: referencio directamente las variables latentes
        List<DiscreteVariable> observedPartitionVars = new ArrayList<>();
        for(DiscreteVariable partitionVar: lcm.getLatentVariables()){
            /*
            DiscreteVariable newManifestVar = new DiscreteVariable(partitionVar.getCardinality(), VariableType.MANIFEST_VARIABLE, partitionVar.getName());
            observedPartitionVars.add(newManifestVar);
            */
            observedPartitionVars.add(partitionVar);
        }

        // Creamos una nueva lista con todas las variables, primero las MVs y luego las LVs (ahora son MV tmb)
        List<DiscreteVariable> completedDataSetVars = new ArrayList<>();
        completedDataSetVars.addAll(dataSet.getVariables());
        completedDataSetVars.addAll(observedPartitionVars);

        // Creamos el dataSet completo que debemos rellenar con los valores de las LVs
        DiscreteData completedDataSet = new DiscreteData(completedDataSetVars);

        // Añadimos las instancias correspondientes
        // Por cada instancia de los datos antiguos añadimos los valores de las MVs seguidos de las LVs
        for(DiscreteDataInstance oldInstance: dataSet.getInstances()){
            int[] oldData = oldInstance.getNumericValues();
            int latentVarData = completeDataMap.get(oldInstance);
            int[] newData = new int[oldData.length + 1];
            // Primero las MVs
            for(int i = 0; i < oldData.length; i++)
                newData[i] = oldData[i];
            // Despues las LVs (en este caso solo hay 1) cuya posicion es la ultima del array
            newData[oldData.length] = latentVarData;

            // Add to the new completed data a new data instance containing both the MVs and the LVs
            completedDataSet.add(new DiscreteDataInstance(newData), dataSet.getWeight(oldInstance));
        }
        return completedDataSet;
    }

    public static DiscreteData generateMultidimensional(DiscreteData dataSet, DiscreteBayesNet mpm) {
        Map<DiscreteDataInstance, List<Integer>> completeDataMap = new HashMap<>();
        List<Tuple<DiscreteDataInstance, List<double[]>>> instancesWithPartitionProbs = AssignToClusters.assignDataCaseToClusters(dataSet, mpm);

        // Transformamos las probabilidades de asignacin de cada instancia a un cluster por el indice del cluster con
        // mayor probabilidad y lo almacenamos
        for (Tuple<DiscreteDataInstance, List<double[]>> instanceWithPartitionProbs : instancesWithPartitionProbs) {

            /** Por cada variable latente del modelo, calculamos el indice del estado con mayor probabilidad*/
            DiscreteDataInstance instance = instanceWithPartitionProbs.getFirst();
            List<Integer> partitionAssignments = new ArrayList<>(mpm.getLatentVariables().size());

            /** Iteramos por el indice de variables latentes */
            for(int partitionVarIndex = 0; partitionVarIndex < mpm.getLatentVariables().size(); partitionVarIndex++){
                double[] partitionProbs = instanceWithPartitionProbs.getSecond().get(partitionVarIndex);
                // Calculamos el índice de valor máximo
                double maxVal = 0;
                int maxIndex = 0;
                for (int i = 0; i < partitionProbs.length; i++)
                    if (partitionProbs[i] > maxVal) {
                        maxIndex = i;
                        maxVal = partitionProbs[i];
                    }
                partitionAssignments.add(maxIndex);
            }

            // Enlazamos cada instancia con su lista de asignaciones para las particiones
            completeDataMap.put(instance, partitionAssignments);
        }
        // Una vez hemos asignado a cada instancia su cluster mas probable, generamos un objeto "DiscreteData" donde
        // se incluyan las variables de particion como variables observadas.

        // Por cada variable latente creamos una MV cuyo nombre coincide y que servirá para el nuevo DataSet
        // TODO 09-07-2018: referencio directamente las variables latentes
        List<DiscreteVariable> observedPartitionVars = new ArrayList<>();
        for(DiscreteVariable partitionVar: mpm.getLatentVariables()){
            /*
            DiscreteVariable newManifestVar = new DiscreteVariable(partitionVar.getCardinality(), VariableType.MANIFEST_VARIABLE, partitionVar.getName());
            observedPartitionVars.add(newManifestVar);
            */
            observedPartitionVars.add(partitionVar);
        }

        // Creamos una nueva lista con todas las variables, primero las MVs y luego las LVs (ahora son MV tmb)
        List<DiscreteVariable> completedDataSetVars = new ArrayList<>();
        completedDataSetVars.addAll(dataSet.getVariables());
        completedDataSetVars.addAll(observedPartitionVars);

        // Creamos el dataSet completo que debemos rellenar con los valores de las LVs
        DiscreteData completedDataSet = new DiscreteData(completedDataSetVars);

        // Añadimos las instancias correspondientes
        // Por cada instancia de los datos antiguos añadimos los valores de las MVs seguidos de las LVs
        for(DiscreteDataInstance oldInstance: dataSet.getInstances()){
            int[] oldData = oldInstance.getNumericValues();
            List<Integer> latentVarData = completeDataMap.get(oldInstance);
            int[] newData = new int[oldData.length + latentVarData.size()];
            for(int i = 0; i < oldData.length; i++)
                newData[i] = oldData[i];
            // Despues las LVs
            for(int i = 0; i < latentVarData.size(); i++)
                newData[i+oldData.length] = latentVarData.get(i);

            // Add to the new completed data a new data instance containing both the MVs and the LVs
            completedDataSet.add(new DiscreteDataInstance(newData), dataSet.getWeight(oldInstance));
        }
        return completedDataSet;
    }
}
