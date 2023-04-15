
import model.DistanceLabel;
import model.KMeans;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

/**
 * Our proposed FPQ algorithm.
 */
public class FPQ {

    private int numSubVectors; // num of sub-vectors
    private int k; // k of k-Means
    private int[] trainLabels;
    private int[][] compressedData;
    private ArrayList<double[][]> subvectorCentroids;
    private double[][] distances;

    private int upperLimit;

    public FPQ(int numSubVectors, int c) {
        this.numSubVectors = numSubVectors;
        this.k = (int) Math.pow(2, c);
        this.distances = new double[this.k][this.numSubVectors];

    }

    public void compress(int[][] trainData, int[] trainLabels) {
        this.upperLimit = 2 * (trainData.length * numSubVectors / k);
        this.trainLabels = trainLabels; // used for prediction
        this.compressedData = new int[trainData.length][this.numSubVectors];
        this.subvectorCentroids = new ArrayList<>();

        int[][][] splitedData = getSubVectors(trainData);

        pKMeans(trainData, splitedData);

        // Generate compressed features
        KMeans[] kMeansArray = new KMeans[splitedData.length];

        for (int i = 0; i < splitedData.length; i++) {
            kMeansArray[i] = new KMeans(trainData.length, this.k);
        }
        System.out.println(trainData.length + "  " + numSubVectors);

        HashMap<Integer, Integer> hashMap = new HashMap<>();
        for (int sampleIdx = 0; sampleIdx < trainData.length; sampleIdx++) {
            for (int compressFeaIdx = 0; compressFeaIdx < numSubVectors; compressFeaIdx++) {
                int[] clusters = kMeansArray[compressFeaIdx].predict2(splitedData[compressFeaIdx][sampleIdx], subvectorCentroids.get(compressFeaIdx));
                int code = clusters[0];
                if(hashMap.containsKey(code)){
                    int num = hashMap.get(code);
                    if(num>upperLimit){
                        code = clusters[1];
                    }
                }

                compressedData[sampleIdx][compressFeaIdx] = code;

                if(hashMap.containsKey(code)){
                    int num = hashMap.get(code) + 1;
                    hashMap.put(code, num);
                }else{
                    hashMap.put(code, 1);
                }
            }
        }
        System.out.println("Compression ends");

        // testing code
//        int sum = 0;
//        for(Map.Entry<Integer, Integer> map:hashMap.entrySet()){
//            System.out.println("Key: " + map.getKey() + " Value: " + map.getValue());
//            sum += map.getValue();
//        }
//        System.out.println("Sum: " + sum);

    }

    /**
     * Split one 1D vector into multiple subvectors,
     */
    private int[][] split2SubVectors(int[] sampleVector) {
        int sampleLen = sampleVector.length;
        int subVecSize = (int) Math.ceil(sampleLen * 1.0 / numSubVectors);
        int[][] subVectors = new int[numSubVectors][subVecSize];

        for (int i = 0; i < numSubVectors; i++) {
            for (int j = 0; j < subVecSize; j++) {
                if (i * subVecSize + j < sampleLen) {
                    subVectors[i][j] = sampleVector[i * subVecSize + j];
                }
            }
        }
        return subVectors;
    }


    public double[] calculateRecallPrecision(int[][] trainData, int[] trainLabels){
        double[] arr = new double[]{0.0, 0.0, 0.0};

        int trueNegative = 0;
        int falsePositive = 0;
        int falseNegative = 0;
        int truePositive = 0;

        for(int i=0; i< trainData.length; i++){
            int predictLabel = predict(trainData[i], 100);

            if(predictLabel == trainLabels[i]){
                if(predictLabel == 0){ // correctly classified as zero
                    truePositive += 1;
                }else{ // correctly classified as non-zero
                    trueNegative += 1;
                }
            }else{
                if(predictLabel == 0){ // incorrectly classified as zero
                    falsePositive += 1;
                }else{ // incorrectly classified as non-zero
                    falseNegative += 1;
                }
            }
        }

        // testing code
//        System.out.println(truePositive);
//        System.out.println(trueNegative);
//        System.out.println(falseNegative);
//        System.out.println(falsePositive);
        arr[0] = (truePositive * 100.0) / (truePositive + falseNegative);
        arr[1] = ((truePositive + trueNegative) * 100.0) / trainLabels.length;
        arr[2] = (falsePositive * 100.0) / (falsePositive + falseNegative);

        return arr;
    }

    private void pKMeans(int[][] trainData, int[][][] splitedData){
        // Parallelized K-means clustering for each set of sub vectors
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        for (int i = 0; i < splitedData.length; i++) {
            int[][] subVectors = splitedData[i];
            System.out.printf("Cluster No. %d out of %d\n", i+1, splitedData.length);
            executor.execute(() -> {
                KMeans km = new KMeans(trainData.length, this.k);
                subvectorCentroids.add(km.fit(subVectors));
            });
        }
        executor.shutdown();
        try {
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
    /**
     * Divide each row of vectors in training data into several sub vectors
     */
    private int[][][] getSubVectors(int[][] trainData) {
        int subVecSize = (int) Math.ceil(trainData[0].length * 1.0 / k);
        // 3d vector, the 2D vector before were divided into n parts
        // trainData.length is the # of training data
        int[][][] trainDataSplit = new int[numSubVectors][trainData.length][subVecSize];

        // process each sample
        for (int sampleIdx = 0; sampleIdx < trainData.length; sampleIdx++) {
            int[] sample = trainData[sampleIdx];

            // 1D vector split into multiple 1D vector
            int[][] subVectors = split2SubVectors(sample);

            // parallel processing of subVectors
            int finalSampleIdx = sampleIdx;
            IntStream.range(0, subVectors.length).parallel().forEach(subIdx -> {
                int[] subVec = subVectors[subIdx];
                trainDataSplit[subIdx][finalSampleIdx] = subVec;
            });
        }

        return trainDataSplit;
    }

    /**
     * Predicts the label based on
     *
     * @param testSample
     * @param nearestNeighbors k
     * @return test image classification (0..9)
     */
    public int predict(int[] testSample, int nearestNeighbors) {

        // calculate distance from each subvector to each clustering centre
        // k is # of clustering centre
        int[][] subVectors = split2SubVectors(testSample);

        for (int subIdx = 0; subIdx < numSubVectors; subIdx++) {
            for (int centerIdx = 0; centerIdx < subvectorCentroids.get(subIdx).length; centerIdx++) {
                this.distances[centerIdx][subIdx] = calculateDistance(subVectors[subIdx], subvectorCentroids.get(subIdx)[centerIdx]);
            }
        }

        // we get distance from each subvector to each clustering centre from last step
        // now each subvector will belong to a clustering centre
        // Calculate the distance to each sample, and use the priority queue to take the label of top k distance
        Queue<DistanceLabel> distanceLabels = new PriorityQueue<>();
        for (int sampleIdx = 0; sampleIdx < compressedData.length; sampleIdx++) {
            double dis = 0;

            for (int feaIdx = 0; feaIdx < this.numSubVectors; feaIdx++) {
                dis += distances[compressedData[sampleIdx][feaIdx]][feaIdx];
            }
            int label = trainLabels[sampleIdx];
            distanceLabels.add(new DistanceLabel(dis, label));
        }

        // Calculate the most frequent tags in the top k neighborhood
        // take top 100 out from distance rankings

        Map<Integer, Integer> labelCnt = new HashMap<>();
        for (int i = 0; i < nearestNeighbors; i++) {
            int label = distanceLabels.poll().label;
            labelCnt.putIfAbsent(label, 0);
            labelCnt.put(label, labelCnt.get(label) + 1);
        }

        int label = 0;
        int maxCnt = 0;
        for (Map.Entry<Integer, Integer> entry : labelCnt.entrySet()) {
            if (entry.getValue() > maxCnt) {
                maxCnt = entry.getValue();
                label = entry.getKey();
            }
        }
        return label;
    }

    private static double calculateDistance(int[] example1, double[] example2) {
        double sum = 0.0;
        for (int i = 0; i < example1.length; ++i) {
            sum += Math.pow(example1[i] - example2[i], 2);
        }
        return sum;
    }

}
