package voltric.io.model.xmlbif;

import org.w3c.dom.CharacterData;
import org.w3c.dom.*;
import voltric.model.DiscreteBayesNet;
import voltric.potential.Function;
import voltric.util.Tuple;
import voltric.variables.DiscreteVariable;
import voltric.variables.modelTypes.VariableType;

import javax.xml.parsers.DocumentBuilderFactory;
import java.io.File;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

/**
 * TODO: Por ahora no permite Latent Variables, habria que incluirlo como una propiedad, pero dado que Weka no trabaja con LVs no es una prioridad.
 * Trabaja con la version de XMLBIF 0.3
 */
public class XmlBifReader {

    public static DiscreteBayesNet processFile(File file) throws Exception{
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        factory.setValidating(true);
        Document doc = factory.newDocumentBuilder().parse(file);
        doc.normalize();

        String bnName = buildName(doc, "XML-BIF net from " + file.getName());
        List<DiscreteVariable> variables = buildVariables(doc, new ArrayList<>());
        DiscreteBayesNet bn = buildStructure(doc, bnName, variables);

        return bn;
    }

    public static DiscreteBayesNet processFile(File file, List<String> latentVariables) throws Exception{
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        factory.setValidating(true);
        Document doc = factory.newDocumentBuilder().parse(file);
        doc.normalize();

        String bnName = buildName(doc, "XML-BIF net from " + file.getName());
        List<DiscreteVariable> variables = buildVariables(doc, latentVariables);
        DiscreteBayesNet bn = buildStructure(doc, bnName, variables);

        return bn;
    }

    public static DiscreteBayesNet processString(String bnXML) throws Exception{
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        factory.setValidating(true);
        Document doc = factory.newDocumentBuilder().parse(
                new org.xml.sax.InputSource(new StringReader(bnXML)));
        doc.normalize();

        String bnName = buildName(doc, "XML-BIF net from string");
        List<DiscreteVariable> variables = buildVariables(doc, new ArrayList<>());
        DiscreteBayesNet bn = buildStructure(doc, bnName, variables);

        return bn;
    }

    public static DiscreteBayesNet processString(String bnXML, List<String> latentVariables) throws Exception{
        DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
        factory.setValidating(true);
        Document doc = factory.newDocumentBuilder().parse(
                new org.xml.sax.InputSource(new StringReader(bnXML)));
        doc.normalize();

        String bnName = buildName(doc, "XML-BIF net from string");
        List<DiscreteVariable> variables = buildVariables(doc, latentVariables);
        DiscreteBayesNet bn = buildStructure(doc, bnName, variables);

        return bn;
    }

    /**
     *
     * @param doc
     * @param defaultName
     * @return
     * @throws Exception
     */
    private static String buildName(Document doc, String defaultName) throws Exception{

        String name = defaultName;

        // Get the name of the network
        NodeList nodelist = selectAllNames(doc);
        if (nodelist.getLength() > 0) {
            name = ((CharacterData) (nodelist.item(0).getFirstChild())).getData();
        }

        return name;
    }

    /**
     * buildInstances parses the BIF document and creates a Bayes Net with its
     * nodes specified, but leaves the network structure and probability tables
     * empty.
     *
     * @param doc DOM document containing BIF document in DOM tree
     * @throws Exception if building fails
     */
    // TODO: Latent variables are always added last
    private static List<DiscreteVariable> buildVariables(Document doc, List<String> latentVariableNames) throws Exception{

        // Get XML variables
        NodeList nodelist = selectAllVariables(doc);
        int nNodes = nodelist.getLength();

        // initialize the list of variables
        List<DiscreteVariable> manifestVariables = new ArrayList<>();
        List<Tuple<String, List<String>>> latentVariables = new ArrayList<>();

        // Process XML variables
        for (int iNode = 0; iNode < nodelist.getLength(); iNode++) {
            // Get element
            ArrayList<Node> valueslist;
            // Get the name of the network
            valueslist = selectOutCome(nodelist.item(iNode));

            int nValues = valueslist.size();
            // generate value strings
            ArrayList<String> nomStrings = new ArrayList<>(nValues + 1);
            for (int iValue = 0; iValue < nValues; iValue++) {
                Node node = valueslist.get(iValue).getFirstChild();
                String sValue = ((CharacterData) (node)).getData();
                if (sValue == null) {
                    sValue = "Value" + (iValue + 1);
                }
                nomStrings.add(sValue);
            }
            ArrayList<Node> nodelist2;
            // Get the name of the network
            nodelist2 = selectName(nodelist.item(iNode));
            if (nodelist2.size() == 0) {
                throw new Exception("No name specified for variable");
            }
            String sNodeName = ((CharacterData) (nodelist2.get(0).getFirstChild())).getData();

            DiscreteVariable var;

            // Separate variables into Latent and Manifest variables
            if(latentVariableNames.contains(sNodeName)) {
                latentVariables.add(new Tuple<>(sNodeName, nomStrings));
            }else {
                var = new DiscreteVariable(sNodeName, nomStrings, VariableType.MANIFEST_VARIABLE);
                manifestVariables.add(var);
            }
        }

        // Manifest variables always go first (see "Cargar BN con IO.md")
        List<DiscreteVariable> variables = new ArrayList<>();
        variables.addAll(manifestVariables);

        // Create latent vasriables now so their index is last
        for(Tuple<String, List<String>> latentVarContent: latentVariables)
            variables.add(new DiscreteVariable(latentVarContent.getFirst(), latentVarContent.getSecond(), VariableType.LATENT_VARIABLE));

        return variables;
    }

    /**
     *
     * @param doc
     * @param bnName
     * @param variables
     * @throws Exception
     */
    private static DiscreteBayesNet buildStructure(Document doc, String bnName, List<DiscreteVariable> variables) throws Exception {
        // First we create an empty BN
        DiscreteBayesNet bn = new DiscreteBayesNet(bnName);

        // Then we add all the variables as nodes
        for(DiscreteVariable var: variables)
            bn.addNode(var);

        for(DiscreteVariable var: variables){
            Element definition = getDefinition(doc, var.getName());
            // Family of variables that will be used to generate a specific CPT
            List<DiscreteVariable> family = new ArrayList<>();

            // Get the parents of the variable
            ArrayList<Node> nodelist = getParentNodes(definition);
            for (int iParent = 0; iParent < nodelist.size(); iParent++) {
                Node parentName = nodelist.get(iParent).getFirstChild();
                String sParentName = ((CharacterData) (parentName)).getData();
                DiscreteVariable parentVar = variables.stream()
                        .filter(x-> x.getName().equals(sParentName))
                        .findFirst()
                        .get();
                // Creates an edge from parent to child in the BN
                bn.addEdge(bn.getNode(var), bn.getNode(parentVar));
                // Adds the parent var to the family of variables
                family.add(parentVar);
            }

            // Finally add the variable (Weka's order)
            family.add(var);

            // Get the CPT
            String sTable = getTable(definition);
            String[] sTableParametersString = sTable.split(" ");

            List<Double> parameters = new ArrayList<>();
            // Eliminate invalid string parameters generated by the split function
            for(int i = 0; i < sTableParametersString.length; i++)
                if(!sTableParametersString[i].equals(""))
                    parameters.add(Double.parseDouble(sTableParametersString[i]));

            // Create aFunction using the order provided by Weka
            Function wekaCpt = Function.createWekaFunction(family, parameters);

            // Assign the new CPT to the variable's BN node
            bn.getNode(var).setCpt(wekaCpt.reorderWekaFunction());
        }

        return bn;
    }

    private static NodeList selectAllNames(Document doc) throws Exception {
        // NodeList nodelist = selectNodeList(doc, "//NAME");
        NodeList nodelist = doc.getElementsByTagName("NAME");
        return nodelist;
    }

    private static NodeList selectAllVariables(Document doc) throws Exception {
        // NodeList nodelist = selectNodeList(doc, "//VARIABLE");
        NodeList nodelist = doc.getElementsByTagName("VARIABLE");
        return nodelist;
    }

    private static ArrayList<Node> selectOutCome(Node item) throws Exception {
        // NodeList nodelist = selectNodeList(item, "OUTCOME");
        ArrayList<Node> nodelist = selectElements(item, "OUTCOME");
        return nodelist;
    }

    private static ArrayList<Node> selectProperty(Node item) throws Exception {
        // NodeList nodelist = selectNodeList(item, "PROPERTY");
        ArrayList<Node> nodelist = selectElements(item, "PROPERTY");
        return nodelist;
    }

    private static ArrayList<Node> selectName(Node item) throws Exception {
        // NodeList nodelist = selectNodeList(item, "NAME");
        ArrayList<Node> nodelist = selectElements(item, "NAME");
        return nodelist;
    }

    private static ArrayList<Node> selectElements(Node item, String sElement) throws Exception {
        NodeList children = item.getChildNodes();
        ArrayList<Node> nodelist = new ArrayList<Node>();
        for (int iNode = 0; iNode < children.getLength(); iNode++) {
            Node node = children.item(iNode);
            if ((node.getNodeType() == Node.ELEMENT_NODE)
                    && node.getNodeName().equals(sElement)) {
                nodelist.add(node);
            }
        }
        return nodelist;
    }

    private static Element getDefinition(Document doc, String sName) throws Exception {
        // NodeList nodelist = selectNodeList(doc,
        // "//DEFINITION[normalize-space(FOR/text())=\"" + sName + "\"]");

        NodeList nodelist = doc.getElementsByTagName("DEFINITION");
        for (int iNode = 0; iNode < nodelist.getLength(); iNode++) {
            Node node = nodelist.item(iNode);
            ArrayList<Node> list = selectElements(node, "FOR");
            if (list.size() > 0) {
                Node forNode = list.get(0);
                if (getContent((Element) forNode).trim().equals(sName)) {
                    return (Element) node;
                }
            }
        }
        throw new Exception("Could not find definition for ((" + sName + "))");
    }

    /**
     * Returns all TEXT children of the given node in one string. Between the node
     * values new lines are inserted.
     *
     * @param node the node to return the content for
     * @return the content of the node
     */
    private static String getContent(Element node) {
        NodeList list;
        Node item;
        int i;
        String result;

        result = "";
        list = node.getChildNodes();

        for (i = 0; i < list.getLength(); i++) {
            item = list.item(i);
            if (item.getNodeType() == Node.TEXT_NODE) {
                result += "\n" + item.getNodeValue();
            }
        }

        return result;
    }

    private static ArrayList<Node> getParentNodes(Node definition) throws Exception {
        // NodeList nodelist = selectNodeList(definition, "GIVEN");
        ArrayList<Node> nodelist = selectElements(definition, "GIVEN");
        return nodelist;
    }

    private static String getTable(Node definition) throws Exception {
        // NodeList nodelist = selectNodeList(definition, "TABLE/text()");
        ArrayList<Node> nodelist = selectElements(definition, "TABLE");
        String sTable = getContent((Element) nodelist.get(0));
        sTable = sTable.replaceAll("\\n", " ");
        return sTable;
    }
}
