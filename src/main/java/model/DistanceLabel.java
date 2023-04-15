package model;

/**
 * to facilitate priority queue
 * by saving distance and label when k-nearest neighbor searches
 */
public class DistanceLabel implements Comparable<DistanceLabel> {
    double dis;
    public int label;

    public DistanceLabel(double dis, int label) {
        this.dis = dis;
        this.label = label;
    }

    @Override
    public int compareTo(DistanceLabel other) {
        return Double.compare(this.dis, other.dis);
    }


}
