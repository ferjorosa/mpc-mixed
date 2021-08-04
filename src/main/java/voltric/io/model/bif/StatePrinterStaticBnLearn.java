package voltric.io.model.bif;

import voltric.potential.util.FunctionIterator;
import voltric.variables.DiscreteVariable;

import java.io.PrintWriter;
import java.util.List;

/**
 * Created by equipo on 06/02/2018.
 */
public class StatePrinterStaticBnLearn implements FunctionIterator.Visitor {

    private PrintWriter writer;

    public StatePrinterStaticBnLearn(PrintWriter writer){
        this.writer = writer;
    }

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
            writer.format("%s", stateName);

            if (i < states.length - 2) {
                writer.write(", ");
            }
        }

        writer.print(") ");
    }

    private void writeEnd() {
        writer.println(";");
    }
}
