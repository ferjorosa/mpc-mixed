package voltric.graph;

import org.apache.commons.lang3.builder.HashCodeBuilder;
import voltric.graph.exception.IllegalEdgeException;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * This class provides an implementation for directed acyclic graphs (DAGs).
 *
 * @author Yi Wang
 * @author ferjorosa
 */
public class DirectedAcyclicGraph<T> extends DirectedGraph<T>{

    /**
     * Default constructor.
     */
    public DirectedAcyclicGraph(){
        super();
    }

    /**
     * Copy constructor. The nodes and edges of this graph are added according to those of the specified graph.
     * However, the graph elements being added are not the same instances as those of the argument graph.
     *
     * @param graph the graph being copied.
     */
    public DirectedAcyclicGraph(DirectedAcyclicGraph<T> graph){
        super(graph);
    }

    /**
     * Adds an edge that connects the two specified nodes to this graph and returns the edge. T
     * here is going to be a run time exception if the resulting graph contains a cycle.
     *
     * @param head head of the edge.
     * @param tail tail of the edge.
     * @return the edge that was added to this graph.
     */
    @Override
    public Edge<T> addEdge(AbstractNode<T> head, AbstractNode<T> tail) {
        // this graph must contain both nodes
        if(!containsNode(head) || !containsNode(tail))
            throw new IllegalArgumentException("The graph must contain both nodes");

        // nodes must be distinct; otherwise, self loop will be introduced.
        if(head.equals(tail))
            throw new IllegalEdgeException("Both nodes must be distinct; otherwise, a self loop will be introduced");

        // nodes cannot be neighbors; otherwise, a duplicated edge will be introduced
        if(head.hasNeighbor(tail))
            throw new IllegalEdgeException("Nodes cannot be neighbours; otherwise either a duplicated edge will be introduced");

        // This graph cannot contain a directed path from the head to the tail; otherwise a directed cycle will be introduced
        if(this.containsPath(head, tail))
            throw new IllegalEdgeException("This graph cannot contain a directed path from the head to the tail; otherwise a directed cycle will be introduced");

        // creates the edge
        Edge<T> edge = new Edge<>(head, tail);

        // adds the edge to the list of edges in this graph
        this.edges.add(edge);

        // attaches the edge to both ends
        head.attachEdge(edge);
        tail.attachEdge(edge);

        ((DirectedNode<T>) head).attachInEdge(edge);
        ((DirectedNode<T>) tail).attachOutEdge(edge);

        return edge;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isEdgeAllowed(AbstractNode<T> head, AbstractNode<T> tail) {
        return super.isEdgeAllowed(head, tail) && !this.containsPath(head, tail);
    }

    public void dfsWithCount(AbstractNode<T> node, Map<AbstractNode<T>, Integer> visitedNodes) {
        // this graph must contain the argument node
        if (!this.containsNode(node))
            throw new IllegalArgumentException("The graph must contain the argument node");

        if (!visitedNodes.containsKey(node))
            visitedNodes.put(node, 1); // discovers the argument node
        else
            visitedNodes.put(node, visitedNodes.get(node) + 1); // rediscovers it

        // explores children
        for (DirectedNode<T> child : ((DirectedNode<T>) node).getChildren()) {
            dfs(child, visitedNodes);
        }
    }

    public void undirectedDfs(AbstractNode<T> node, Map<AbstractNode<T>, Integer> visitedNodes){
        // this graph must contain the argument node
        if(!this.containsNode(node))
            throw new IllegalArgumentException("The graph must contain the argument node");

        visitedNodes.put(node, 1); // discovers the argument node

        // explores unvisited neighbors (sort of an undirected approach)
        for (AbstractNode<T> neighbor : node.getNeighbors()) {
            if (!visitedNodes.containsKey(neighbor)) {
                undirectedDfs(neighbor, visitedNodes);
            }
        }
    }

    // TODO: Needs to be Tested
    public boolean isTree(){
        // 1 - The number of edges in a tree must be number of nodes minus 1
        if (this.getNumberOfEdges() != this.getNumberOfNodes() - 1)
            return false;

        // 2 - A tree can have only one root
        List<DirectedNode<T>> roots = this.getDirectedNodes().stream().filter(x-> x.isRoot()).collect(Collectors.toList());
        if (roots.size() != 1)
            return false;

        // a Depth First Search is performed
        HashMap<AbstractNode<T>, Integer> visitedNodes = new HashMap<>();
        this.dfsWithCount(roots.get(0), visitedNodes);

        // 4 - Each node can only be visited once
        for(int i: visitedNodes.values())
            if(i > 1)
                return false;

        return true;
    }

    // TODO: Needs to be Tested
    public boolean isForest() {
        // A forest can have multiple roots (nodes without parents)
        List<DirectedNode<T>> roots = this.getDirectedNodes().stream().filter(x-> x.isRoot()).collect(Collectors.toList());

        // 1 - The number of edges in a tree must be equal to (nNodes - nRoots)
        if (this.getNumberOfEdges() != this.getNumberOfNodes() - roots.size())
            return false;

        // 2 - DFS is applied from each root and the results are combined in a single Map
        Map<AbstractNode<T>, Integer> combinedVisits = new HashMap<>();
        for(int i=0; i < roots.size(); i++) {

            // 2.1 - DFS from the specific root
            Map<AbstractNode<T>, Integer> visitedNodes = new HashMap<>();
            this.dfsWithCount(roots.get(i), visitedNodes);

            // 2.2 - Combination of the results
            for(Map.Entry<AbstractNode<T>, Integer> e : visitedNodes.entrySet())
                if(!combinedVisits.containsKey(e.getKey()))
                    combinedVisits.put(e.getKey(), e.getValue());
                else
                    combinedVisits.put(e.getKey(), combinedVisits.get(e.getKey()) + e.getValue());
        }

        // 3 - Each node can only be visited once (all nodes should have been visited if isRoot() is working correctly)
        for(int i: combinedVisits.values())
            if(i > 1)
                return false;

        return true;
    }

    // TODO: Needs to be Tested
    public boolean isPolyTree() {

        if(isPolyForest()){
            // 1 - Start an undirected DFS from a node (it doesnt matter which one)
            Map<AbstractNode<T>, Integer> visitedNodes = new HashMap<>();
            undirectedDfs(this.nodes.get(0), visitedNodes);

            // 2 - All nodes the nodes must be connected to the origin
            return visitedNodes.size() == this.getNumberOfNodes();
        }

        return false;
    }

    // TODO: Needs to be Tested
    public boolean isPolyForest() {
        // A polyforest can have multiple roots (nodes without parents)
        List<DirectedNode<T>> roots = this.getDirectedNodes().stream().filter(x-> x.isRoot()).collect(Collectors.toList());

        // 1 - The number of edges in a tree must be equal to (nNodes - nRoots)
        if (this.getNumberOfEdges() != this.getNumberOfNodes() - roots.size())
            return false;

        // 2 - DFS is applied from each root
        for(int i=0; i < roots.size(); i++) {

            // 2.1 - DFS from the specific root
            Map<AbstractNode<T>, Integer> visitedNodes = new HashMap<>();
            this.dfsWithCount(roots.get(i), visitedNodes);

            // 2.2 - Each node reached from the root can only be visited once
            for(int nVisits: visitedNodes.values())
                if(nVisits > 1)
                    return false;
        }

        return true;
    }

    /**
     * Returns the moral graph of this graph. I assume that you know what is a
     * moral graph. Otherwise, you are not supposed to use this method :-)
     *
     * @return the moral graph of this graph.
     */
    public final UndirectedGraph<T> computeMoralGraph() {
        UndirectedGraph<T> moralGraph = new UndirectedGraph<>();

        // copies nodes in this graph
        for (AbstractNode<T> node : this.nodes) {
            moralGraph.addNode(node.getContent());
        }

        // copies edges in this graph with directions dropped
        for (Edge<T> edge : this.edges) {
            moralGraph.addEdge(moralGraph.getNode(edge.getHead().getContent()),
                    moralGraph.getNode(edge.getTail().getContent()));
        }

        // connects nodes that are divorced parents of some node in this DAG.
        for (AbstractNode<T> node : this.nodes) {
            DirectedNode<T> dNode = (DirectedNode<T>) node;

            for (DirectedNode<T> parent1 : dNode.getParents()) {
                AbstractNode<T> neighbor1 = moralGraph.getNode(parent1.getContent());

                for (DirectedNode<T> parent2 : dNode.getParents()) {
                    AbstractNode<T> neighbor2 = moralGraph.getNode(parent2.getContent());

                    if (neighbor1 != neighbor2 && !neighbor1.hasNeighbor(neighbor2)) {
                        moralGraph.addEdge(neighbor1, neighbor2);
                    }
                }
            }
        }

        return moralGraph;
    }

    /**
     * Returns {@code true} if the object is a {@code DirectedAcyclicGraph} with equal fields (inherited ones included).
     *
     * @param object the object to test equality against.
     * @return true if {@code object} equals this.
     */
    @Override
    public boolean equals(Object object){
        if(this == object)
            return true;

        if(object.getClass() != this.getClass())
            return  false;

        DirectedAcyclicGraph graph = (DirectedAcyclicGraph) object;
        return this.nodes.equals(graph.nodes)
                && this.edges.equals(graph.edges)
                && this.contents.equals(graph.contents);
    }

    /**
     * Returns the object's hashcode.
     *
     * @return the object's hashcode.
     */
    @Override
    public int hashCode() {
        return new HashCodeBuilder(3, 23)
                .append(nodes)
                .append(edges)
                .append(uniqueID)
                .toHashCode();
    }

    /**
     * Creates and returns a deep copy of this graph. This implementation copies everything in this graph. Consequently,
     * it is safe to do anything you want to the deep copy.
     *
     * @return a deep copy of this graph.
     */
    @Override
    public DirectedAcyclicGraph<T> clone() {
        return new DirectedAcyclicGraph<>(this);
    }

    /** {@inheritDoc} */
    @Override
    public String toString(int amount) {
        // amount must be non-negative
        if(amount <= 0)
            throw new IllegalArgumentException("The amount must be positive");

        // prepares white space for indent
        StringBuffer whiteSpace = new StringBuffer();
        for (int i = 0; i < amount; i++) {
            whiteSpace.append("\t");
        }

        // builds string representation
        StringBuffer stringBuffer = new StringBuffer();

        stringBuffer.append(whiteSpace);
        stringBuffer.append("directed acyclic graph {\n");

        stringBuffer.append(whiteSpace);
        stringBuffer.append("\tnumber of nodes = " + getNumberOfNodes() + ";\n");

        stringBuffer.append(whiteSpace);
        stringBuffer.append("\tnodes = {\n");

        for (AbstractNode node : this.nodes) {
            stringBuffer.append(node.toString(amount + 2));
        }

        stringBuffer.append(whiteSpace);
        stringBuffer.append("\t};\n");

        stringBuffer.append(whiteSpace);
        stringBuffer
                .append("\tnumber of edges = " + getNumberOfEdges() + ";\n");

        stringBuffer.append(whiteSpace);
        stringBuffer.append("\tedges = {\n");

        for (Edge<T> edge : this.edges) {
            stringBuffer.append(edge.toString(amount + 2));
        }

        stringBuffer.append(whiteSpace);
        stringBuffer.append("\t};\n");

        stringBuffer.append(whiteSpace);
        stringBuffer.append("};");

        return stringBuffer.toString();
    }
}
