package voltric.io.model.bif;

import voltric.data.DiscreteData;
import voltric.learning.score.LearningScore;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.potential.Function;
import voltric.potential.util.FunctionIterator;
import voltric.variables.DiscreteVariable;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * Created by equipo on 06/02/2018.
 */
public class StaticBnLearnBifFileWriter {

    public static void write(OutputStream output, DiscreteBayesNet network, DiscreteData data, String encoding, boolean useTableFormat) throws UnsupportedEncodingException{
        PrintWriter writer = new PrintWriter(new OutputStreamWriter(output, encoding));
        writeNetworkDeclaration(network, writer);
        writeVariables(network, writer);
        writeProbabilities(network, writer, useTableFormat);
        if (data != null)
            writeScore(network, data, writer);
        writer.close();
    }

    public static void write(String filePath, DiscreteBayesNet network) throws FileNotFoundException, UnsupportedEncodingException {
        write(filePath, network, null);
    }

    public static void write(String filePath, DiscreteBayesNet network, DiscreteData data) throws FileNotFoundException, UnsupportedEncodingException {
       write(new FileOutputStream(filePath), network, data, "UTF-8", false);
    }

    /**
     * Writes the network declaration.
     *
     * @param network network to write.
     */
    private static void writeNetworkDeclaration(DiscreteBayesNet network, PrintWriter writer) {
        writer.format("network %s {\n}\n", network.getName());
        writer.println();
    }

    /**
     * Writes the variables part.
     *
     * @param network
     *            network to write.
     */
    private static void writeVariables(DiscreteBayesNet network, PrintWriter writer) {
        List<DiscreteBeliefNode> nodes = network.getNodes();
        for (DiscreteBeliefNode node : nodes)
            writeNode(node, writer);

    }

    /**
     * Writes the information of a belief node.
     *
     * @param node
     *            node to write.
     */
    private static void writeNode(DiscreteBeliefNode node, PrintWriter writer) {
        List<String> states = node.getVariable().getStates();

        writer.format("variable %s {\n", node.getName());

        // write the states
        writer.format("\ttype discrete[%d] { ", states.size());
        for (int i = 0; i < states.size(); i++) {
            writer.format("%s", states.get(i));
            if (i != states.size() - 1)
                writer.print(", ");
            else
                writer.print(" ");
        }
        writer.println("};");

        writer.println("}");
        writer.println();
    }

    /**
     * Writes the probabilities definition part.
     *
     * @param network
     *            network to write.
     */
    private static void writeProbabilities(DiscreteBayesNet network, PrintWriter writer, boolean useTableFormat) {
        List<DiscreteBeliefNode> nodes = network.getNodes();
        for (DiscreteBeliefNode node : nodes) {
            writeProbabilities(node, writer, useTableFormat);
        }
    }

    /**
     * Writes the probabilities definition for a belief node.
     *
     * @param node
     *            node to write.
     */
    private static void writeProbabilities(DiscreteBeliefNode node, PrintWriter writer, boolean useTableFormat) {
        Function function = node.getCpt();

        ArrayList<DiscreteVariable> variables = new ArrayList<>();
        variables.add(node.getVariable());
        List<DiscreteVariable> parentVariables = (node.getParentNodes().stream().map(x->(DiscreteVariable) x.getVariable())).collect(Collectors.toList());
        variables.addAll(parentVariables);

        // write the related variables
        writer.format("probability ( %s", variables.get(0).getName());

        // check if it has parent variables
        if (variables.size() > 1) {
            writer.print(" | ");
            for (int i = 1; i < variables.size(); i++) {
                writer.format("%s", variables.get(i).getName());
                if (i != variables.size() - 1) {
                    writer.print(", "); //
                }
            }
        }

        writer.println(" ) {");

        if (useTableFormat)
            writeProbabilitiesTable(function, variables, writer);
        else
            writeProbabilitiesWithStates(function, variables, writer);

        writer.println("}");
        writer.println();
    }

    private static void writeProbabilitiesTable(Function function, ArrayList<DiscreteVariable> variables, PrintWriter writer) {
        double[] cells = function.getCells(variables);
        writer.print("\ttable ");
        for (int i = 0; i < cells.length; i++) {
            writer.print(cells[i]);
            if (i != cells.length - 1) {
                writer.print(", ");
            }
        }
        writer.println(";");
    }

    private static void writeProbabilitiesWithStates(Function function, ArrayList<DiscreteVariable> variables, PrintWriter writer) {
        // use table format for root variable
        if (variables.size() == 1) {
            writeProbabilitiesTable(function, variables, writer);
            return;
        }

        // put the parent variables at the beginning for iteration
        ArrayList<DiscreteVariable> order = new ArrayList<>(variables.size());
        for (int i = 1; i < variables.size(); i++)
            order.add(variables.get(i));
        order.add(variables.get(0));

        FunctionIterator iterator = new FunctionIterator(function, order);
        iterator.iterate(new StatePrinterStaticBnLearn(writer));
    }

    private static void writeScore(DiscreteBayesNet network, DiscreteData data, PrintWriter writer) {
        writer.println();
        writer.format("//Loglikelihood: %f\n", LearningScore.calculateLogLikelihood(data, network));
        writer.format("//BIC Score: %f\n", LearningScore.calculateBIC(data, network));
        writer.println();
    }
}
