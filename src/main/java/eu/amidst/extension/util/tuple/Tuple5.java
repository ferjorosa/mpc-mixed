package eu.amidst.extension.util.tuple;

/**
 * Generic tuple implementation for 4 objects
 */
public class Tuple5<A, B, C, D, E> {

    private A first;

    private B second;

    private C third;

    private D fourth;

    private E fifth;


    public Tuple5(A first, B second, C third, D fourth, E fifth) {
        super();
        this.first = first;
        this.second = second;
        this.third = third;
        this.fourth = fourth;
        this.fifth = fifth;
    }

    public A getFirst() {
        return first;
    }

    public B getSecond() {
        return second;
    }

    public C getThird() {
        return third;
    }

    public D getFourth() {
        return fourth;
    }

    public E getFifth() {
        return fifth;
    }

    public boolean equals(Object other) {

        if (other == null)
            return false;

        if (other instanceof Tuple5) {
            Tuple5 otherTuple5 = (Tuple5) other;

            if(this.first.equals(otherTuple5.first)
                    && this.second.equals(otherTuple5.second)
                    && this.third.equals(otherTuple5.third)
                    && this.fourth.equals(otherTuple5.fourth)
                    && this.fifth.equals(otherTuple5.fifth)
            )
                return true;
        }

        return false;
    }

    public int hashCode() {
        int hashFirst = this.first != null ? this.first.hashCode() : 0;
        int hashSecond = this.second != null ? this.second.hashCode() : 0;
        int hashThird = this.third != null ? this.third.hashCode() : 0;
        int hashFourth = this.fourth != null ? this.fourth.hashCode() : 0;
        int hashFifth = this.fifth != null ? this.fifth.hashCode() : 0;

        return (hashFirst + hashSecond + hashThird + hashFourth + hashFifth) * 7;
    }

    public String toString() {
        return "(" + this.first + ", " + this.second + ", " + this.third + "," + this.fourth + "," + this.fifth + ")";
    }
}
