package model;

/**
 * Copyright (c) KU Leuven - All rights reserved.
 */
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;
import java.util.Set;

public class KMeans {
    private Set<Integer> indices;
    private int n_centroids;

    public KMeans(int am_examples, int n_centroids) {
        this.indices = this.initializeCentroidsIndices(am_examples, n_centroids);
        this.n_centroids = n_centroids;
    }

    private Set<Integer> initializeCentroidsIndices(int am_examples, int n_centroids) {
        Random randomGenerator = new Random(42L);
        Set<Integer> indices = new HashSet();

        while(indices.size() != n_centroids) {
            int index;
            for(index = randomGenerator.nextInt(am_examples); indices.contains(index); index = randomGenerator.nextInt(am_examples)) {
            }

            indices.add(index);
        }

        return indices;
    }

    public double[][] fit(int[][] traindata) {
        int features = traindata[0].length;
        double[][] centroids = new double[this.n_centroids][features];
        int a = 0;

        for(Iterator var6 = this.indices.iterator(); var6.hasNext(); ++a) {
            int index = (Integer)var6.next();

            for(int j = 0; j < features; ++j) {
                centroids[a][j] = (double)traindata[index][j];
            }
        }

        boolean converged = false;

        while(!converged) {
            HashMap<Integer, ArrayList<int[]>> clusters = this.assignToClusters(traindata, centroids);
            double[][] newCentroids = this.updateCentroids(clusters, centroids);
            boolean change = false;

            label38:
            for(int i = 0; i < centroids.length; ++i) {
                for(int j = 0; j < centroids[0].length; ++j) {
                    if (centroids[i][j] != newCentroids[i][j]) {
                        change = true;
                        break label38;
                    }
                }
            }

            if (!change) {
                converged = true;
            } else {
                centroids = newCentroids;
            }
        }

        return centroids;
    }

    private double[][] updateCentroids(HashMap<Integer, ArrayList<int[]>> clusters, double[][] centroids) {
        double[][] newCentroids = new double[centroids.length][centroids[0].length];

        int cluster;
        double[] avg;
        for(Iterator var5 = clusters.keySet().iterator(); var5.hasNext(); newCentroids[cluster] = avg) {
            cluster = (Integer)var5.next();
            ArrayList<int[]> examples = (ArrayList)clusters.get(cluster);
            avg = new double[centroids[0].length];
            Iterator var9 = examples.iterator();

            while(var9.hasNext()) {
                int[] e = (int[])var9.next();

                for(int i = 0; i < avg.length; ++i) {
                    avg[i] += (double)e[i];
                }
            }

            for(int i = 0; i < avg.length; ++i) {
                avg[i] /= (double)examples.size() * 1.0;
            }
        }

        return newCentroids;
    }

    private HashMap<Integer, ArrayList<int[]>> assignToClusters(int[][] traindata, double[][] centroids) {
        HashMap<Integer, ArrayList<int[]>> clusters = new HashMap();
        int[][] var8 = traindata;
        int var7 = traindata.length;

        for(int var6 = 0; var6 < var7; ++var6) {
            int[] example = var8[var6];
            int cluster = this.predict(example, centroids);
            if (!clusters.containsKey(cluster)) {
                clusters.put(cluster, new ArrayList());
            }

            ((ArrayList)clusters.get(cluster)).add(example);
        }

        return clusters;
    }

    public int predict(int[] example, double[][] centroids) {
        int cluster = 0;
        double minDistance = calculateDistance(example, centroids[0]);

        for(int i = 1; i < centroids.length-1; ++i) {
            double newDistance = calculateDistance(example, centroids[i]);
            if (minDistance > newDistance) {
                cluster = i;
                minDistance = newDistance;
            }
        }

        return cluster;
    }

    // new
    public int[] predict2(int[] example, double[][] centroids) {
        int[] cluster = new int[]{0,0};
        double minDistance = calculateDistance(example, centroids[0]);

        for(int i = 1; i < centroids.length; ++i) {
            double newDistance = calculateDistance(example, centroids[i]);
            if (minDistance > newDistance) {
                cluster[1] = cluster[0];
                cluster[0] = i;
                minDistance = newDistance;
            }
        }

        return cluster;
    }

    private static double calculateDistance(int[] example1, double[] example2) {
        double sum = 0.0;

        for(int i = 0; i < example1.length; ++i) {
            sum += Math.pow((double)example1[i] - example2[i], 2.0);
        }

        return Math.sqrt(sum);
    }
}

