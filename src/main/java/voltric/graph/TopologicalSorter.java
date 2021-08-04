package voltric.graph;

import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedList;
import java.util.List;

/**
 * Esta clase devuelve las lista de nodos ordenados de forma topologica.  Cuando haga una transformacion del codigo
 * se podria añadir este codigo a Voltric, en la clase DAG
 */
public class TopologicalSorter {

    public static <T> List<DirectedNode<T>> sort(DirectedAcyclicGraph<T> dag) {

        Deque<DirectedNode<T>> sortedNodes = new LinkedList<>(); // JDK Queue
        List<DirectedNode<T>> visitedNodes = new ArrayList<>();

        /* Iteramos por el conjunto de nodos del grafo, el punto de inicio es indiferente para el algoritmo */
        for(DirectedNode<T> node: dag.getDirectedNodes()){
            if(!visitedNodes.contains(node))
                visit(node, visitedNodes, sortedNodes);
        }

        return (List<DirectedNode<T>>) sortedNodes; // Given its underlying is a LinkedList, we simply cast it to List

    }

    private static <T> void visit(DirectedNode<T> node, List<DirectedNode<T>> visitedNodes, Deque<DirectedNode<T>> sortedNodes) {

        visitedNodes.add(node);

        // explores unvisited children
        for (DirectedNode<T> child :  node.getChildren()) {
            if (!visitedNodes.contains(child)) {
                visit(child, visitedNodes, sortedNodes);
            }
        }

        // Lo añadimos al final del metodo para que este la lista correctamente ordenada
        sortedNodes.addFirst(node);
    }
}
