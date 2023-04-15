import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;


public class Experiment {

    // you may modify the training/testing data amount here
    public static final int trainDataLength = 50000;
    public static final int testDataLength = 5000;

    /**
     * According to the suggestion from PQ paper, we use
     * k* = 256 (2^8) and m=8 (code length, i.e. int) in this program.
     */
    public static void main(String[] args) throws InterruptedException {

        System.out.println("Experiment 1 (PQ): \n");
        experiment1(30, 8);
        System.out.println("----------------");
        System.out.println("Experiment 2 (FPQ): \n");
        experiment2(30, 8);

    }

    // PQ
    private static void experiment1(int n, int c) throws InterruptedException{
        long start = System.currentTimeMillis();
        PQ pq = new PQ(n, c);

        // Read data
        String trainPath = Experiment.class.getClassLoader().getResource("mnist_train.csv").getFile();
        DataReader.DataSet trainData = DataReader.readData(trainPath, trainDataLength);

        // Compress
        pq.compress(trainData.data, trainData.labels);

        long end = System.currentTimeMillis();
        System.out.printf("\nCompressed data elapsed time %.2f seconds\n\n", (end - start) / 1000.0);

        // Recalculate prediction time consumption
        start = System.currentTimeMillis();
        String testPath = Experiment.class.getClassLoader().getResource("mnist_test.csv").getFile();
        DataReader.DataSet testData = DataReader.readData(testPath, testDataLength);

        System.gc();
        Thread.sleep(1000);
        MemoryMXBean memoryMXBean = ManagementFactory.getMemoryMXBean();
        System.out.println("Heap Memory Usage: " + memoryMXBean.getHeapMemoryUsage() + "\n");

        // correct prediction #
        int right = 0;

        for (int i = 0; i < testDataLength; i++) {
            // make prediction
            int predLabel = pq.predict(testData.data[i], 100);

            if (predLabel == testData.labels[i]) {
                right += 1;
            }

        }
        end = System.currentTimeMillis();
        System.out.printf("Predict elapsed time %.2f seconds\n\n", (end - start) / 1000.0);

        System.out.println("Calculate Accuracy, Recall, Precision and Fallout...");
        System.out.printf("Accuracy: %.2f%%\n", right * 100.0 / testDataLength);
        double[] arr = pq.calculateRecallPrecision(trainData.data, trainData.labels);
        System.out.printf("Recall: %.2f%%\nPrecision: %.2f%%\nFall-out: %.2f%%\n", arr[0], arr[1], arr[2]);

    }

    //FPQ
    static void experiment2(int n, int c) throws InterruptedException{

        long start = System.currentTimeMillis();
        FPQ FPQ = new FPQ(n,c);

        // Read data
        String trainPath = Experiment.class.getClassLoader().getResource("mnist_train.csv").getFile();
        DataReader.DataSet trainData = DataReader.readData(trainPath, trainDataLength);

        // Compress
        FPQ.compress(trainData.data, trainData.labels);

        long end = System.currentTimeMillis();
        System.out.printf("\nCompressed data elapsed time %.2f seconds\n\n", (end - start) / 1000.0);

        // calculate prediction and time consumption
        start = System.currentTimeMillis();
        String testPath = Experiment.class.getClassLoader().getResource("mnist_test.csv").getFile();
        DataReader.DataSet testData = DataReader.readData(testPath, testDataLength);

        System.gc();
        Thread.sleep(1000);
        MemoryMXBean memoryMXBean = ManagementFactory.getMemoryMXBean();
        System.out.println("Heap Memory Usage: " + memoryMXBean.getHeapMemoryUsage() + "\n");

        // total num of testData
        int total = 0;
        // correct prediction #
        int right = 0;
        for (int i = 0; i < testData.data.length; i++) {
            // make prediction
            int predLabel = FPQ.predict(testData.data[i], 100);
            total += 1;
            if (predLabel == testData.labels[i]) {
                right += 1;
            }
        }
        end = System.currentTimeMillis();
        System.out.printf("Predict elapsed time %.2f seconds\n\n", (end - start) / 1000.0);
        System.out.println("Calculate Accuracy, Recall, Precision and Fallout...");
        System.out.printf("Accuracy: %.2f%%\n", right * 100.0 / testDataLength);
        double[] arr = FPQ.calculateRecallPrecision(trainData.data, trainData.labels);
        System.out.printf("Recall: %.2f%%\nPrecision: %.2f%%\nFall-out: %.2f%%\n", arr[0], arr[1], arr[2]);
    }
}
