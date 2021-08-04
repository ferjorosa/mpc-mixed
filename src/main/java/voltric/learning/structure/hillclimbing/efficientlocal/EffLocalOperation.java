package voltric.learning.structure.hillclimbing.efficientlocal;

import voltric.variables.DiscreteVariable;

import java.util.Objects;

public class EffLocalOperation {

    public enum Type {
        OPERATION_ADD,
        OPERATION_DEL,
        OPERATION_REVERSE
    }

    private DiscreteVariable tailVar;

    private DiscreteVariable headVar;

    private double score;

    private Type type;

    public EffLocalOperation(DiscreteVariable headVar, DiscreteVariable tailVar, double score, Type type) {
        this.tailVar = tailVar;
        this.headVar = headVar;
        this.score = score;
        this.type = type;
    }

    public DiscreteVariable getTailVar() {
        return tailVar;
    }

    public DiscreteVariable getHeadVar() {
        return headVar;
    }

    public double getScore() {
        return score;
    }

    public Type getType() {
        return type;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        EffLocalOperation that = (EffLocalOperation) o;
        return Objects.equals(tailVar, that.tailVar) &&
                Objects.equals(headVar, that.headVar) &&
                type == that.type;
    }

    @Override
    public int hashCode() {
        return Objects.hash(tailVar, headVar, type);
    }
}
