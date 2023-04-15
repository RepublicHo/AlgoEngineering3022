import java.util.*;
import model.DistanceLabel;
import model.KMeans;

/**
 * PQ algorithm
 */
public class PQ {

    private int numSubVectors; // num of sub-vectors
    private int k; // k of k-Means
    private int[] trainLabels;
    private int[][] compressedData;
    private ArrayList<double[][]> subvectorCentroids;
    private double[][] distances;

    public PQ(int numSubVectors, int c) {

        this.numSubVectors = numSubVectors;
        this.k = (int) Math.pow(2, c);
    }

    public void compress(int[][] trainData, int[] trainLabels) {
        this.trainLabels = trainLabels; // used for prediction
        this.compressedData = new int[trainData.length][this.numSubVectors];
        this.subvectorCentroids = new ArrayList<>();

        int[][][] trainDataSplited = getSubVectors(trainData);

        // 2d vector splitted into multiple 2d vector, each one represents one part of data
        // now cluster each 2d vector, get cluster centre
        for (int i = 0; i < trainDataSplited.length; i++) {

            int[][] subVectors = trainDataSplited[i];
            System.out.printf("Cluster No. %d out of %d\n", i+1, trainDataSplited.length);
            KMeans km = new KMeans(trainData.length, this.k);
            subvectorCentroids.add(km.fit(subVectors));
            // testing code:
//            System.out.println(km.fit(subVectors)[0][0]);
//            System.out.println(i + " ends");
        }

        for (int sampleIdx = 0; sampleIdx < trainData.length; sampleIdx++) {
            for (int compressFeaIdx = 0; compressFeaIdx < numSubVectors; compressFeaIdx++) {
                KMeans km = new KMeans(trainData.length, this.k);
                int code = km.predict(trainDataSplited[compressFeaIdx][sampleIdx], subvectorCentroids.get(compressFeaIdx));
                compressedData[sampleIdx][compressFeaIdx] = code;

                // testing code
//                System.out.println("Code: " + code + " sampleIndex: " + sampleIdx + " compressFeatureIndex: " + compressFeaIdx);
            }
        }
        // testing code
        System.out.println("Compression ends");
    }

    private static double calculateDistance(int[] example1, double[] example2) {
        double sum = 0.0;
        for (int i = 0; i < example1.length; ++i) {
            sum += Math.pow(example1[i] - example2[i], 2);
        }
        return sum;
    }

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


    private int[][][] getSubVectors(int[][] trainData) {
        int subVecSize = (int) Math.ceil(trainData[0].length * 1.0 / k);

        // 3d vector, the 2D vector before were divided into n parts
        int[][][] trainDataSplited = new int[numSubVectors][trainData.length][subVecSize];

        // process each sample
        for (int sampleIdx = 0; sampleIdx < trainData.length; sampleIdx++) {
            int[] sample = trainData[sampleIdx];

            // 1D vector split into multiple 1D vector
            int[][] subVectors = split2SubVectors(sample);
            for (int subIdx = 0; subIdx < subVectors.length; subIdx++) {
                int[] subVec = subVectors[subIdx];
                trainDataSplited[subIdx][sampleIdx] = subVec;
            }
        }
        return trainDataSplited;
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

    public int predict(int[] testSample, int nearestNeighbors) {
        this.distances = new double[this.k][this.numSubVectors];

        // after sub-vectors obtained from splitting, calculate distance
        // from each sub-vector to each clustering centre
        int[][] subVectors = split2SubVectors(testSample);
        for (int subIdx = 0; subIdx < numSubVectors; subIdx++) {
            for (int centerIdx = 0; centerIdx < subvectorCentroids.get(subIdx).length; centerIdx++) {
                this.distances[centerIdx][subIdx] = calculateDistance(subVectors[subIdx], subvectorCentroids.get(subIdx)[centerIdx]);
            }
        }


        // each subvector will belong to a clustering centre
        // Calculate the distance to each sample, and use the priority queue
        // to take the label of top k distance
        Queue<DistanceLabel> distanceLabels = new PriorityQueue<>();
        for (int sampleIdx = 0; sampleIdx < compressedData.length; sampleIdx++) {
            double dis = 0;
            for (int feaIdx = 0; feaIdx < this.numSubVectors; feaIdx++) {
                dis += distances[compressedData[sampleIdx][feaIdx]][feaIdx];
            }
            int label = trainLabels[sampleIdx];

            // testing code
//            System.out.println("label: " + label + " distance: " + dis);
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

}
