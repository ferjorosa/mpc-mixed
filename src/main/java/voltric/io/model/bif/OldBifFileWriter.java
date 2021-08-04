package voltric.io.model.bif;

import voltric.graph.DirectedNode;
import voltric.model.DiscreteBayesNet;
import voltric.model.DiscreteBeliefNode;
import voltric.variables.DiscreteVariable;
import voltric.variables.Variable;

import java.io.*;
import java.util.ArrayList;
import java.util.Date;
import java.util.Iterator;

/**
 * Los archivos OBIF deberian acabar en .obif
 *
 * TODO: Falta la property que define si una variable es latente o manifest
 *
 * NOTA 26-04-2018: No lee bien las tablas, utilizar XMLBifReader en vez de el, aunque sea peor
 */
@Deprecated
public class OldBifFileWriter {

    public static void writeBif(String filePath, DiscreteBayesNet bayesNet)throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter out = new PrintWriter(new BufferedWriter(
                new OutputStreamWriter(new FileOutputStream(filePath), "UTF8")));

        // outputs header
        out.println("// " + filePath);
        out.println("// Produced at "+ (new Date(System.currentTimeMillis())));

        // outputs name
        out.println("network \"" + bayesNet.getName() + "\" {");
        out.println("}");
        out.println();

        // outputs nodes
        for (DiscreteBeliefNode node : bayesNet.getNodes()) {
            DiscreteVariable variable = node.getVariable();

            // name of variable
            if(variable.isLatentVariable())
                out.println("variable \"" + "LV_" + variable.getName() + "\" {");
            else
                out.println("variable \"" + "MV_" + variable.getName() + "\" {");

            // states of variable
            out.print("\ttype discrete[" + variable.getCardinality() + "] { ");
            Iterator<String> iter = variable.getStates().iterator();
            while (iter.hasNext()) {
                out.print("\"" + iter.next() + "\"");
                if (iter.hasNext()) {
                    out.print(" ");
                }
            }
            out.println(" };");

            out.println("}");
            out.println();
        }

        // Output CPTs
        for (DiscreteBeliefNode node : bayesNet.getNodes()) {

            // variables in this family. note that the variables in the
            // probability block are arranged from the most significant place to
            // the least significant place.
            ArrayList<DiscreteVariable> vars = new ArrayList<DiscreteVariable>();
            vars.add(node.getVariable());

            // name of node
            if(node.getContent().isManifestVariable())
                out.print("probability ( \"" +"MV_" +node.getName() + "\" ");
            else if(node.getContent().isLatentVariable())
                out.print("probability ( \"" +"LV_" +node.getName() + "\" ");

            // names of parents
            if (!node.isRoot()) {
                out.print("| ");
            }

            Iterator<DirectedNode<Variable>> iter = node.getParents().iterator();
            while (iter.hasNext()) {
                DiscreteBeliefNode parent = (DiscreteBeliefNode) iter.next();

                if(parent.getContent().isManifestVariable())
                    out.print("\"" + "MV_" + parent.getName() + "\"");
                else if(parent.getContent().isLatentVariable())
                    out.print("\"" + "LV_"+ parent.getName() + "\"");

                if (iter.hasNext()) {
                    out.print(", ");
                }

                vars.add(parent.getVariable());
            }
            out.println(" ) {");

            // cells in CPT
            out.print("\ttable");
            for (double cell : node.getCpt().getCells(vars)) {
                out.print(" " + cell);
            }
            out.println(";");
            out.println("}");
        }

        out.close();
    }
}
