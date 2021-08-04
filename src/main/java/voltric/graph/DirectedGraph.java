package voltric.graph;

import org.apache.commons.lang3.builder.HashCodeBuilder;
import voltric.graph.exception.IllegalEdgeException;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * This class provides an implementation for directed graphs.
 *
 * @author Yi Wang
 * @author ferjorosa
 */
public class DirectedGraph<T> extends AbstractGraph<T> {

    /**
     * Default constructor.
     */
    public DirectedGraph(){
        super();
    }

    /**
     * Copy constructor. The nodes and edges of this graph are added according to those of the specified graph.
     * However, the graph elements being added are not the same instances as those of the argument graph.
     *
     * @param graph the graph being copied.
     */
    //TODO: Revisar, un mismo "interfaz" de copia (forma de copiar)
    public DirectedGraph(DirectedGraph<T> graph){
        super(graph);
    }

    /** {@inheritDoc} */
    @Override
    public DirectedNode<T> addNode(AbstractNode<T> node) {
        if(!(node instanceof DirectedNode))
            throw new IllegalArgumentException("Only directed nodes are allowed");

        // adds the node to the list of nodes in this graph
        this.nodes.add(node);

        // maps name to node
        this.contents.put(node.getContent(), node);

        return (DirectedNode<T>) node;
    }

    /** {@inheritDoc} */
    @Override
    public DirectedNode<T> addNode(T content) {

        // the node's content must be unique in this graph
        if(this.containsNode(content))
            throw new IllegalArgumentException("Node content objects must be unique");

        // creates node
        DirectedNode<T> node = new DirectedNode<>(this, content);

        // adds the node to the list of nodes in this graph
        this.nodes.add(node);

        // maps name to node
        this.contents.put(content, node);

        return node;
    }

    /** {@inheritDoc} */
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

        // creates the edge
        Edge<T> edge = new Edge<>(head, tail);

        // adds the edge to the list of edges in this graph
        this.edges.add(edge);

        // attaches the edge from both ends (adds nodes as each others neighbours)
        head.attachEdge(edge);
        tail.attachEdge(edge);

        ((DirectedNode<T>) head).attachInEdge(edge);
        ((DirectedNode<T>) tail).attachOutEdge(edge);

        return edge;
    }

    /** {@inheritDoc} */
    @Override
    public void removeEdge(Edge<T> edge) {
        // this graph must contain the argument edge
        if(!this.containsEdge(edge))
            throw new IllegalArgumentException("The graph must contain the argument edge");

        // removes edge from the list of edges in this graph
        this.edges.remove(edge);

        // detaches the edge from both ends (removes nodes as each others neighbours)
        DirectedNode<T> edgeHead = (DirectedNode<T>) edge.getHead();
        DirectedNode<T> edgeTail = (DirectedNode<T>) edge.getTail();

        edgeHead.detachInEdge(edge);
        edgeTail.detachOutEdge(edge);
        //edge.getHead().detachEdge(edge);
        //edge.getTail().detachEdge(edge);

        //((DirectedNode<T>) edge.getHead()).detachInEdge(edge);
        //((DirectedNode<T>) edge.getTail()).detachOutEdge(edge);
    }

    /**
     * Removes the specified edge from this graph.
     *
     * @param head the head of the edge.
     * @param tail the tail of the edge.
     */
    public void removeEdge(AbstractNode<T> head, AbstractNode<T> tail) {
        Optional<Edge<T>> possibleEdge = this.getEdge(head, tail);
        if(!possibleEdge.isPresent())
            throw new IllegalArgumentException("Edge doesn't exist");
        else
            this.removeEdge(possibleEdge.get());
    }

    /** {@inheritDoc} */
    @Override
    public boolean containsEdge(AbstractNode<T> head, AbstractNode<T> tail) {

        // this graph must contain both nodes
        if(!containsNode(head) || !containsNode(tail))
            throw new IllegalArgumentException("The graph must contain both nodes");

        // creates the edge
        Edge<T> edge = new Edge<>(head, tail);

        return this.edges.contains(edge);
    }

    /** {@inheritDoc} */
    @Override
    public boolean isEdgeAllowed(AbstractNode<T> head, AbstractNode<T> tail) {

        /** this graph must contain both nodes */
        if(!containsNode(head) || !containsNode(tail))
            return false;

        /** nodes must be distinct; otherwise, self loop will be introduced. */
        if(head.equals(tail))
            return false;

        /** nodes cannot be neighbors; otherwise, a duplicated edge will be introduced */
        if(head.hasNeighbor(tail))
            return false;

        /** Temporally creates the edge, but without the necessary substructures, just to check the Map */
        Edge<T> edge = new Edge<>(head, tail);

        return !this.edges.contains(edge);
    }

    /** {@inheritDoc} */
    @Override
    public DirectedNode<T> getNode(T content) {
        return (DirectedNode<T>) super.getNode(content);
    }

    /**
     * Returns the list of nodes in this graph. For the sake of efficiency, this implementation returns the reference to
     * the protected field. Make sure you understand this before using this method.
     *
     * <p><b>Note:</b> Given the lack of COVARIANCE in the Java's collection library, this method cannot be properly
     * overridden. That is why {@code getDirectedNodes()} exists.</p>
     *
     * @return the list of nodes in this graph.
     *
     * @see #getDirectedNodes()
     */
    @Override
    public List<AbstractNode<T>> getNodes() {
        return super.getNodes();
    }

    /**
     * Returns a List containing the graph's casted nodes.
     *
     * @return a List containing the graph's casted nodes.
     */
    public List<DirectedNode<T>> getDirectedNodes(){
        return this.getNodes().stream().map(x-> (DirectedNode<T>) x).collect(Collectors.toList());
    }

    @Override
    public void dfs(AbstractNode<T> node, Map<AbstractNode<T>, Integer> visitedNodes) {
        // this graph must contain the argument node
        if(!this.containsNode(node))
            throw new IllegalArgumentException("The graph must contain the argument node");

        visitedNodes.put(node, 1); // discovers the argument node

        // explores unvisited children
        for (DirectedNode<T> child : ((DirectedNode<T>) node).getChildren()) {
            if (!visitedNodes.containsKey(child)) {
                dfs(child, visitedNodes);
            }
        }
    }

    /**
     * Reverses the specific edge.
     *
     * @param edge the edge to be reversed.
     */
    public Edge<T> reverseEdge(Edge<T> edge){
        this.removeEdge(edge);
        return this.addEdge(edge.getHead(), edge.getTail());
    }

    /**
     * Reverses the directions of the edges and returns the resulting directed graph.
     *
     * @return	graph with directions of edges reversed
     */
    public DirectedGraph<T> reverseEdges() {
        DirectedAcyclicGraph<T> result = new DirectedAcyclicGraph<>();

        // copies nodes
        for (AbstractNode<T> node : this.nodes) {
            result.addNode(node.getContent());
        }

        // copies edges
        for (Edge<T> edge : this.edges) {
            AbstractNode<T> originalHead = result.getNode(edge.getHead().getContent());
            AbstractNode<T> originalTail = result.getNode(edge.getTail().getContent());

            // reverse the edge
            result.addEdge(originalTail, originalHead);
        }

        return result;
    }

    /**
     * Returns an undirected graph based on this graph by ignoring the directions of the edges.
     *
     * @return	undirected version of this graph.
     */
    public UndirectedGraph<T> getUndirectedGraph() {
        UndirectedGraph<T> undirectedGraph = new UndirectedGraph<>();

        // copies nodes
        for (AbstractNode<T> node : this.nodes) {
            undirectedGraph.addNode(node.getContent());
        }

        // copies edges
        for (Edge<T> edge : this.edges) {
            undirectedGraph.addEdge(
                    undirectedGraph.getNode(edge.getHead().getContent()),
                    undirectedGraph.getNode(edge.getTail().getContent()));
        }

        return undirectedGraph;
    }

    /**
     * Returns {@code true} if the object is a {@code DirectedGraph} with equal fields (inherited ones included).
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

        DirectedGraph graph = (DirectedGraph) object;
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
        return new HashCodeBuilder(5, 3)
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
    public DirectedGraph<T> clone() {
        return new DirectedGraph<>(this);
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
        stringBuffer.append("directed graph {\n");

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
        stringBuffer.append("\tnumber of edges = " + getNumberOfEdges() + ";\n");

        stringBuffer.append(whiteSpace);
        stringBuffer.append("\tedges = {\n");

        for (Edge edge : this.edges) {
            stringBuffer.append(edge.toString(amount + 2));
        }

        stringBuffer.append(whiteSpace);
        stringBuffer.append("\t};\n");

        stringBuffer.append(whiteSpace);
        stringBuffer.append("};");

        return stringBuffer.toString();
    }
}
