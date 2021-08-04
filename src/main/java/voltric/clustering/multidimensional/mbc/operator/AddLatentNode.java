package voltric.clustering.multidimensional.mbc.operator;

import org.apache.commons.math3.util.Combinations;
import voltric.data.DiscreteData;
import voltric.graph.Edge;
import voltric.learning.LearningResult;
import voltric.learning.structure.latent.StructuralEM;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.variables.DiscreteVariable;
import voltric.variables.Variable;
import voltric.variables.modelTypes.VariableType;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Añade un nodo latente entre el par de MVs que devuelve el score mas alto. El proceso de añadir un nodo latente
 * se puede realizar de multiples formas. En este caso no vamos a añadir nodos de la manera en la que lo haciamos para
 * los OLCMs con {@link voltric.clustering.multidimensional.olcm.OlcmHillClimbing}, sino que nos vamos a inspirar en el
 * trabajo de chen et al. (2012).
 *
 * La introduccion de un nodo LV se realiza por tanto SOLO ENTRE MVs, la razon de esto es porque tiene sentido a nivel
 * de generar particiones que agrupen atributos y mediante arcos ver como dichas particiones se relacionan entre si. En
 * el caso de contar con muchas particiones (por tanto muchas LVs) si que quizas tendria sentido agrupar las variables
 * latentes de forma jerarquica.
 *
 * Proceso:
 *
 * IMPORTANTE: Al introducir un nodo latente, si las MVs pertenecen a una o mas particiones, dicha LV pasara a tener como
 * padres el conjunto de variables que era padre de alguna de las dos MVs seleccionadas. En el PEOR caso posible, dicha
 * variable latente podria tener 2 * k padres, siendo k el numero maximo de padres k. Es el unico caso donde se superaria,
 * lo que seguramente ocurra internamente es que si añade muchos padres, generara muchos parametros y por tanto se pondra
 * a eliminar aquellos que considere innecesarios.
 *
 *  - TODO: Este es un aspecto MUY IMPORTANTE, ya que genera parametros de forma exponencial, lo que puede hacer que lo limitemos
 *
 * IMPORTANTE: Los arcos existentes entre las MVs seleccionadas ya sea entre si o con otras MVs se mantienen como parte
 * de la estructura inicial que será pasada al Structural EM.
 *
 * La introduccion de un nodo latente se produce mediante la seleccion de dos MVs. Dichas dos MVs dejaran de pertenecer a
 * las LV con las que estuviesen relacionadas y pasaran a estarlo unicamente con la nuevo (arcos de bridge, mantienen los
 * arcos entre features). Cualquier par de MVs es seleccionable a menos que:
 *
 * - Al quitar  dos variables del bridge de una latente esta se quede sin hijos (pasaria a tener unicamente la nueva
 * LV como hija, lo cual la hace redundante). Es decir, dicha LV debe tener 3 o mas hijos.
 *
 * NOTA: Dado que cada vez que introducimos un nuevo nodo se llama al SEM, se pueden formar estructuras diferentes y por lo
 * tanto no nos basta con eliminar la nueva LV y devolver las MVs a su particion anterior, por ello es necesario
 * clonar la red inicial por cada nueva LV, de esa forma podemos automaticamente restablecer el estado inicial.
 *
 * Sin embargo, es necesario actualizar las restricciones del SEM, ya que sino podria añadir arcos MV -> LV
 */
public class AddLatentNode implements LatentMbcHcOperator {

    private int variableCardinality;

    public AddLatentNode(int variableCardinality) {
        this.variableCardinality = variableCardinality;
    }

    @Override
    public LearningResult<DiscreteBayesNet> apply(DiscreteBayesNet seedNet, DiscreteData data, StructuralEM sem) {

        /*
            No clonamos la red al inicio como en los operadores OLHC ya que cada cambio en la variable latente tiene
            consecuencias en la estructura que serian costosos de revertir al final de cada iteracion, de esta manera
            es mucho mas simple.
         */
        double bestModelScore = -Double.MAX_VALUE;
        LearningResult<DiscreteBayesNet> bestModelResult = null;

        /* Generamos el conjunto combinaciones de pares de MVs utilizando Apache Math. Cada combinacion se corresponde con los indices */
        Iterator<int[]> combinationsOfPairMVs = new Combinations(seedNet.getManifestVariables().size(), 2).iterator();

        /* Se itera por cada par de MVs, se selecciona aquellos pares que no dejen a alguna de las LVs sin hijos */
        while(combinationsOfPairMVs.hasNext()) {
            int[] combinations = combinationsOfPairMVs.next();
            int mv1 = combinations[0]; // Indice de la primera MV
            int mv2 = combinations[1]; // Indice de la segunda MV

            /* Comprobamos que no sean iguales */
            if(mv1 != mv2) {

                DiscreteVariable firstMV = seedNet.getManifestVariables().get(mv1);
                DiscreteVariable secondMV = seedNet.getManifestVariables().get(mv2);


                /* Si tienen algun padre en comun, este debe tener 3 o mas hijos */

                // 1 - Generamos la lista de padres latentes de cada variable
                Set<Variable> firstMvLatentParents = seedNet.getNode(firstMV).getParents()
                        .stream()
                        .map(x-> x.getContent())
                        .filter(x->x.isLatentVariable())
                        .collect(Collectors.toSet());
                Set<Variable> secondMvLatentParents = seedNet.getNode(secondMV).getParents()
                        .stream()
                        .map(x-> x.getContent())
                        .filter(x->x.isLatentVariable())
                        .collect(Collectors.toSet());

                // 2 - Generamos la lista de padres comunes
                List<Variable> commonParents = new ArrayList<>();
                for(Variable firstMvParent: firstMvLatentParents) {
                    for (Variable secondMvParent : secondMvLatentParents) {
                        if(firstMvParent.equals(secondMvParent))
                            commonParents.add(firstMvParent);
                    }
                }

                // 3 - Comprobamos que todos los padres comunes tengan al menos 3 hijos
                boolean commonParentsHave3ChildrenOrMore = true;
                for(Variable parent: commonParents) {
                    if(seedNet.getNode(parent).getChildren().size() < 3)
                        commonParentsHave3ChildrenOrMore = false;
                }

                    /*
                        Una vez sabemos que los dos nodos cumplen las condiciones para ser elegibles, copiamos la red inicial
                        y creamos un nodo latente cuyos hijos son las MVs seleccionadas.

                        Dichas MVs pasaran a formar parte UNICA Y EXCLUSIVAMENTE de la variable latente seleccionada, pero
                        esta seguira recibiendo la influencia de las otras LVs que eran padres de "firstMV" y "secondMV" al
                        añadir un arco de estas a la nueva LV.
                     */
                if(commonParentsHave3ChildrenOrMore) {

                    // 1 - Copiamos la red inicial para no modificarla y poder volver a empezar de cero con cada iteracion
                    DiscreteBayesNet clonedNet = seedNet.clone();

                    // 2 - Creamos un nodo latente cuyos 2 hijos son "firstMV" y "secondMV"
                    DiscreteVariable newLatentVar = new DiscreteVariable(this.variableCardinality, VariableType.LATENT_VARIABLE);
                    DiscreteBeliefNode newLatentNode = clonedNet.addNode(newLatentVar);
                    clonedNet.addEdge(clonedNet.getNode(firstMV), newLatentNode);
                    clonedNet.addEdge(clonedNet.getNode(secondMV), newLatentNode);

                    // 3 - Eliminamos los arcos de los padres latentes a dichas variables
                    for(Variable firstMvParent: firstMvLatentParents){
                        Edge<Variable> edge = clonedNet.getEdge(clonedNet.getNode(firstMV), clonedNet.getNode(firstMvParent)).get();
                        clonedNet.removeEdge(edge);
                    }

                    for(Variable secondMvParent: secondMvLatentParents){
                        Edge<Variable> edge = clonedNet.getEdge(clonedNet.getNode(secondMV), clonedNet.getNode(secondMvParent)).get();
                        clonedNet.removeEdge(edge);
                    }

                    // 4 - Añadimos un arcos de cada uno de los padres latentes de la primera MV a la nueva LV
                    for(Variable firstMvParent: firstMvLatentParents)
                        clonedNet.addEdge(newLatentNode, clonedNet.getNode(firstMvParent));

                    // 5 - Filtramos el conjunto de los padres de la segunda variable con los padres comunes (para no repetir)
                    Set<Variable> uniqueSecondParents = secondMvLatentParents.stream().filter(x->!commonParents.contains(x)).collect(Collectors.toSet());
                    for(Variable secondMvParent: uniqueSecondParents)
                        clonedNet.addEdge(newLatentNode, clonedNet.getNode(secondMvParent));

                    // 6 - Aprendemos los parametros de este modelo con el EM
                    LearningResult<DiscreteBayesNet> initialModelForSEM = sem.getEm().learnModel(clonedNet, data);

                    // 7 - Añadimos restricciones al SEM para la nueva variable latente
                    sem.addLatentVar(newLatentVar);

                    LearningResult<DiscreteBayesNet> semResult;

                    // 8 - Tomamos el modelo aprendido como punto de inicio para el Structural EM
                    semResult = sem.learnModel(initialModelForSEM, data);

                    // 9 - Una vez aprendido el modelo con SEM, comprobamos que mejore el score actual y si es asi, lo almacenamos
                    if (semResult.getScoreValue() > bestModelScore) {
                        bestModelScore = semResult.getScoreValue();
                        bestModelResult = semResult;
                    }

                    // 10 - Una vez se ha comprobado si el modelo mejora o no el score actual, se pasa a la siguiente iteracion
                    // Aunque no se revierten cambios en la estructura por utilizar una copia en cada iteracion, es
                    // necesario eliminar las restricciones del SEM para la nueva LV que habiamos añadido
                    sem.removeLatentVar(newLatentVar);
                }
            }
        }
        /*
        for(DiscreteVariable firstMV: seedNet.getManifestVariables()) {
            for (DiscreteVariable secondMV : seedNet.getManifestVariables()) {
                if (!firstMV.equals(secondMV)) {

                    // 1 - Generamos la lista de padres latentes de cada variable
                    Set<Variable> firstMvLatentParents = seedNet.getNode(firstMV).getParents()
                            .stream()
                            .map(x-> x.getContent())
                            .filter(x->x.isLatentVariable())
                            .collect(Collectors.toSet());
                    Set<Variable> secondMvLatentParents = seedNet.getNode(secondMV).getParents()
                            .stream()
                            .map(x-> x.getContent())
                            .filter(x->x.isLatentVariable())
                            .collect(Collectors.toSet());

                    // 2 - Generamos la lista de padres comunes
                    List<Variable> commonParents = new ArrayList<>();
                    for(Variable firstMvParent: firstMvLatentParents) {
                        for (Variable secondMvParent : secondMvLatentParents) {
                            if(firstMvParent.equals(secondMvParent))
                                commonParents.add(firstMvParent);
                        }
                    }

                    // 3 - Comprobamos que todos los padres comunes tengan al menos 3 hijos
                    boolean commonParentsHave3ChildrenOrMore = true;
                    for(Variable parent: commonParents) {
                        if(seedNet.getNode(parent).getChildren().size() < 3)
                            commonParentsHave3ChildrenOrMore = false;
                    }

                    if(commonParentsHave3ChildrenOrMore) {

                        // 1 - Copiamos la red inicial para no modificarla y poder volver a empezar de cero con cada iteracion
                        DiscreteBayesNet clonedNet = seedNet.clone();

                        // 2 - Creamos un nodo latente cuyos 2 hijos son "firstMV" y "secondMV"
                        DiscreteVariable newLatentVar = new DiscreteVariable(this.variableCardinality, VariableType.LATENT_VARIABLE);
                        DiscreteBeliefNode newLatentNode = clonedNet.addNode(newLatentVar);
                        clonedNet.addEdge(clonedNet.getNode(firstMV), newLatentNode);
                        clonedNet.addEdge(clonedNet.getNode(secondMV), newLatentNode);

                        // 3 - Eliminamos los arcos de los padres latentes a dichas variables
                        for(Variable firstMvParent: firstMvLatentParents){
                            Edge<Variable> edge = clonedNet.getEdge(clonedNet.getNode(firstMV), clonedNet.getNode(firstMvParent)).get();
                            clonedNet.removeEdge(edge);
                        }

                        for(Variable secondMvParent: secondMvLatentParents){
                            Edge<Variable> edge = clonedNet.getEdge(clonedNet.getNode(secondMV), clonedNet.getNode(secondMvParent)).get();
                            clonedNet.removeEdge(edge);
                        }

                        // 4 - Añadimos un arcos de cada uno de los padres latentes de la primera MV a la nueva LV
                        for(Variable firstMvParent: firstMvLatentParents)
                            clonedNet.addEdge(newLatentNode, clonedNet.getNode(firstMvParent));

                        // 5 - Filtramos el conjunto de los padres de la segunda variable con los padres comunes (para no repetir)
                        Set<Variable> uniqueSecondParents = secondMvLatentParents.stream().filter(x->!commonParents.contains(x)).collect(Collectors.toSet());
                        for(Variable secondMvParent: uniqueSecondParents)
                            clonedNet.addEdge(newLatentNode, clonedNet.getNode(secondMvParent));

                        // 6 - Aprendemos los parametros de este modelo con el EM
                        LearningResult<DiscreteBayesNet> initialModelForSEM = sem.getEm().learnModelWithPriorUpdate(clonedNet, data);

                        // 7 - Añadimos restricciones al SEM para la nueva variable latente
                        sem.addLatentVar(newLatentVar);

                        // 8 - Tomamos el modelo aprendido como punto de inicio para el Structural EM
                        LearningResult<DiscreteBayesNet> semResult = sem.learnModelWithPriorUpdate(initialModelForSEM, data);

                        // 9 - Una vez aprendido el modelo con SEM, comprobamos que mejore el score actual y si es asi, lo almacenamos
                        if (semResult.getScoreValue() > bestModelScore) {
                            bestModelScore = semResult.getScoreValue();
                            bestModelResult = semResult;
                        }

                        // 10 - Una vez se ha comprobado si el modelo mejora o no el score actual, se pasa a la siguiente iteracion
                        // Aunque no se revierten cambios en la estructura por utilizar una copia en cada iteracion, es
                        // necesario eliminar las restricciones del SEM para la nueva LV que habiamos añadido
                        sem.removeLatentVar(newLatentVar);
                    }
                }
            }
        }

        */

        /* Finalmente si el modelo ha sido modificado, lo devolvemos */
        if(bestModelResult != null)
            return bestModelResult;

        /* En caso contrario, devolvemos un modelo "falso" con score infinitamente malo */
        return new LearningResult<>(null, bestModelScore, sem.getScoreType());
    }
}
