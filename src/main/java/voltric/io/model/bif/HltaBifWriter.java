package voltric.io.model.bif;

import voltric.data.DiscreteData;
import voltric.learning.score.LearningScore;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.potential.Function;
import voltric.potential.util.FunctionIterator;
import voltric.variables.DiscreteVariable;

import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.List;

/**
 * Writes Bayesian networks in BIF format.
 *
 * Version HLTA que escribe en un formato entendible por bnlearn.
 * TODO: No escribe bien las probabilidades en formato state, hay que revisar el orden de los parents y la manera en que
 * itera las celdas de la Funciton.
 *
 * @author leonard
 * 
 */
@Deprecated
public class HltaBifWriter {

	/**
	 * Constructs this writer with an underlying output stream, using the
	 * default UTF-8 encoding.
	 * 
	 * @param output
	 *            output stream where the network is written to.
	 * @throws UnsupportedEncodingException
	 */
	public HltaBifWriter(OutputStream output) throws UnsupportedEncodingException {
		this(output, false, "UTF-8");
	}

	/**
	 * Constructs this writer with an underlying output stream, using the
	 * default UTF-8 encoding.
	 * 
	 * @param output
	 *            output stream where the network is written to.
	 * @param useTableFormat
	 *            whether to use table format in probability definition
	 * @throws UnsupportedEncodingException
	 */
	public HltaBifWriter(OutputStream output, boolean useTableFormat) throws UnsupportedEncodingException {
		this(output, useTableFormat, "UTF-8");
	}

	/**
	 * Constructs this writer with an underlying output stream.
	 * 
	 * @param output
	 *            output stream where the network is written to.
	 * @param useTableFormat
	 *            whether to use table format in probability definition
	 * @param encoding
	 *            charset used for the output.
	 * @throws UnsupportedEncodingException
	 */
	public HltaBifWriter(OutputStream output, boolean useTableFormat, String encoding) throws UnsupportedEncodingException {
		this.useTableFormat = useTableFormat;
		writer = new PrintWriter(new OutputStreamWriter(output, encoding));
	}

	/**
	 * Writes the network.
	 */
	public void write(DiscreteBayesNet network) {
		writeNetworkDeclaration(network);
		writeVariables(network);
		writeProbabilities(network);
		writer.close();
	}

	/**
	 * Writes the network declaration.
	 * 
	 * @param network
	 *            network to write.
	 */
	private void writeNetworkDeclaration(DiscreteBayesNet network) {
		writer.format("network \"%s\" {\n}\n", network.getName());
		writer.println();
	}

	/**
	 * Writes the variables part.
	 * 
	 * @param network
	 *            network to write.
	 */
	private void writeVariables(DiscreteBayesNet network) {
		List<DiscreteBeliefNode> nodes = network.getNodes();
		for (DiscreteBeliefNode node : nodes) {
			DiscreteBeliefNode beliefNode = (DiscreteBeliefNode) node;
			writeNode(beliefNode);
		}
	}

	/**
	 * Writes the information of a belief node.
	 * 
	 * @param node
	 *            node to write.
	 */
	private void writeNode(DiscreteBeliefNode node) {
		List<String> states = node.getVariable().getStates();

		writer.format("variable \"%s\" {\n", node.getName());

		// write the states
		writer.format("\ttype discrete[%d] { ", states.size());
		for (int i = 0; i < states.size() - 1; i++) {
			writer.format("\"%s\", ", states.get(i));
		}
		writer.format("\"%s\", ", states.get(states.size() - 1)); // Last one without comma

		writer.println("};");

		writer.println("}");
		writer.println();
	}

	/**
	 * Writes the probabilities definition part.
	 * 
	 * @param network network to write.
	 */
	private void writeProbabilities(DiscreteBayesNet network) {
		List<DiscreteBeliefNode> nodes = network.getNodes();
		for (DiscreteBeliefNode node : nodes) {
			writeProbabilities(node);
		}
	}

	/**
	 * Writes the probabilities definition for a belief node.
	 * 
	 * @param node
	 *            node to write.
	 */
	private void writeProbabilities(DiscreteBeliefNode node) {
		Function function = node.getCpt();

		List<DiscreteVariable> variables = node.getCpt().getVariables();

		// write the related variables
		writer.format("probability (\"%s\"", variables.get(0).getName());

		// check if it has parent variables
		if (variables.size() > 1) {
			writer.print(" | ");
			for (int i = 1; i < variables.size(); i++) {
				writer.format("\"%s\"", variables.get(i).getName());
				if (i != variables.size() - 1) {
					writer.print(", ");
				}
			}
		}

		writer.println(") {");

		if (useTableFormat)
			writeProbabilitiesTable(function, variables);
		else
			writeProbabilitiesWithStates(function, variables);

		writer.println("}");
		writer.println();
	}

	private void writeProbabilitiesTable(Function function, List<DiscreteVariable> variables) {
		double[] cells = function.getCells(variables);
		writer.print("\ttable ");
		for (int i = 0; i < cells.length; i++) {
			writer.print(cells[i]);
			if (i != cells.length - 1) {
				writer.print(" ");
			}
		}
		writer.println(";");
	}

	private void writeProbabilitiesWithStates(Function function, List<DiscreteVariable> variables) {
		// use table format for root variable
		if (variables.size() == 1) {
			writeProbabilitiesTable(function, variables);
			return;
		}

		// put the parent variables at the beginning for iteration
		ArrayList<DiscreteVariable> order = new ArrayList<>(variables.size());
		for (int i = 1; i < variables.size(); i++)
			order.add(variables.get(i));
		order.add(variables.get(0));

		FunctionIterator iterator = new FunctionIterator(function, order);
		iterator.iterate(new StateVisitor());
	}

	private void writeScore(DiscreteBayesNet network, DiscreteData data) {
		writer.println();
		writer.format("//Loglikelihood: %f\n", LearningScore.calculateLogLikelihood(data, network));
		writer.format("//BIC Score: %f\n", LearningScore.calculateBIC(data, network));
		writer.println();
	}

	/**
	 * The print writer encapsulating the underlying output stream.
	 */
	private final PrintWriter writer;

	private boolean useTableFormat = true;

	private class StateVisitor implements FunctionIterator.Visitor {
		public void visit(List<DiscreteVariable> order, int[] states, double value) {
			// the node state and variable (instead of parent variables)
			int nodeState = states[states.length - 1];
			DiscreteVariable nodeVariable = order.get(states.length - 1);

			if (nodeState == 0) {
				writeStart(order, states);
			}

			writer.print(value);

			if (nodeState == nodeVariable.getCardinality() - 1) {
				writeEnd();
			} else {
				writer.print(", ");
			}
		}

		private void writeStart(List<DiscreteVariable> order, int[] states) {
			writer.print("\t(");
			// write parent states, which excludes the last state
			for (int i = 0; i < states.length - 1; i++) {
				String stateName = order.get(i).getStates().get(states[i]);
				writer.format("\"%s\"", stateName);

				if (i < states.length - 2) {
					writer.write(" ");
				}
			}

			writer.print(") ");
		}

		private void writeEnd() {
			writer.println(";");
		}
	}
}
