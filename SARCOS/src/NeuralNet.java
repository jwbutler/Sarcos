import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Random;
import java.util.Scanner;
public class NeuralNet {
  private ArrayList<InputNode> inputNodes;
  private ArrayList<MiddleNode> middleNodes;
  private OutputNode outputNode;
  private int numColumns;
  private double avg, stdDev;
  private double RSQ;
  private final boolean NORMALIZE = true;
  private final int MIDDLE_NODES = 7;
  private final int TRAINING_ITERATIONS = 2000;
  private final double LEARNING_RATE = 0.00003;
  private final double K = 6; // logistic curve parameter
  private final double NORM_DIVISOR = 100;
  
  public static void main(String[] args) {
    new NeuralNet();
  }
  
  public NeuralNet() {
    numColumns = 21;
    //handValidate();
    run();
  }
  
  public void run() {
    constructNetwork();
    ArrayList<Double[]> trainData = readData("Sarcos_Data1_train.csv");
    ArrayList<Double> trainTargets = readTargets("Sarcos_Data1_train.csv", false);
    ArrayList<Double[]> testData = readData("Sarcos_Data1_test.csv");
    ArrayList<Double> testTargets = readTargets("Sarcos_Data1_test.csv", true);
    train(trainData, trainTargets, testData, testTargets);
    test(testData, testTargets);
  }

  public void handValidate() {
    /* Load the train data and partition it into two sets. */
    ArrayList<Double[]> trainData = readData("Sarcos_Data1_train.csv");
    ArrayList<Double> trainTargets = readTargets("Sarcos_Data1_train.csv", false);
    ArrayList<Double[]> testData = new ArrayList<Double[]>();
    ArrayList<Double> testTargets = new ArrayList<Double>();
    Random RNG = new Random();
    for (int i=0; i<trainData.size()/2; i++) {
      int j = RNG.nextInt(trainData.size());
      testData.add(trainData.remove(j));
      testTargets.add(trainTargets.remove(j));
    }
    constructNetwork();
    train(trainData, trainTargets, testData, testTargets);
    test(testData, testTargets);
  }
  
  public ArrayList<Double[]> readData(String filename) {
    ArrayList<Double[]> data = new ArrayList<Double[]>();
    try {
      Scanner sc = new Scanner(new File(filename));
      sc.nextLine(); // skip the header line
      while (sc.hasNextLine()) {
        String line = sc.nextLine();
        String[] toks = line.split(",");
        Double[] row = new Double[toks.length-1];
        for (int i=2; i<toks.length; i++) {
          row[i-2] = Double.parseDouble(toks[i]); 
        }
        data.add(row);
      }
      sc.close();
    } catch (FileNotFoundException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
    return data;
  }

  public ArrayList<Double> readTargets(String filename, boolean test) {
    ArrayList<Double> targets = new ArrayList<Double>();
    double min=0.0, max=0.0;
    try {
      Scanner sc = new Scanner(new File(filename));
      sc.nextLine(); // skip the header line
      while (sc.hasNextLine()) {
        String line = sc.nextLine();
        String[] toks = line.split(",");
        double target = Double.parseDouble(toks[1]);
        if (!test) avg += target;
        /* EXPERIMENTAL: NORMALIZE TARGET */ 
        targets.add(target);
        if (!test && target<min) min=target;
        if (!test && target>max) max=target;
      }
      if (NORMALIZE) {
        if (!test) avg /= targets.size();
        stdDev = 0.0;
        for (double t : targets) {
          stdDev += Math.pow(t - avg, 2);
        }
        stdDev = Math.sqrt(stdDev/targets.size());
        for (int i=0; i<targets.size(); i++) {
          targets.set(i, normalize(targets.get(i)));
        }
      }
      sc.close();
    } catch (FileNotFoundException e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
    return targets;
  }
  
  public void constructNetwork() {
    inputNodes = new ArrayList<InputNode>();
    middleNodes = new ArrayList<MiddleNode>();
    for (int i=0; i<numColumns; i++) {
      inputNodes.add(new InputNode());
    }
    for (int i=0; i<MIDDLE_NODES; i++) {
      middleNodes.add(new MiddleNode(inputNodes, MiddleNode.LOG2, K));
    }
    outputNode = new OutputNode(middleNodes, OutputNode.LINEAR);
  }
  
  public void updateNodes(ArrayList<Double[]> data, int index) {
    for (int i=0; i<inputNodes.size(); i++) {
      inputNodes.get(i).data = data.get(index)[i];
    }
  }
  
  public void updateWeights(ArrayList<Double[]> data, double target) {
    outputNode.updateWeights(target, LEARNING_RATE);
    for (MiddleNode n: middleNodes) {
      n.updateWeights(target, LEARNING_RATE);
    }
  }
  
  public void train(ArrayList<Double[]> trainData, ArrayList<Double> trainTargets, ArrayList<Double[]> testData, ArrayList<Double> testTargets) {
    for (int i=0; i<TRAINING_ITERATIONS; i++) {
      /*if (i%25 == 0) {
        testQuiet(testData, testTargets);
        System.out.printf("Training iteration #%d, RSQ = %f\n", i, RSQ);
      }*/
      for (int j=0; j<trainData.size(); j++) {
        updateNodes(trainData, j);
        updateWeights(trainData, trainTargets.get(j));
      }
    }
  }
  
  public void test(ArrayList<Double[]> data, ArrayList<Double> targets) {
    RSQ = 0;
    double prediction, target = 0;
    for (int i=0; i<data.size(); i++) {
      updateNodes(data, i);
      if (NORMALIZE) {
        prediction = deNormalize(outputNode.calculate());
        target = deNormalize(targets.get(i));
      } else {
        prediction = outputNode.calculate();
        target = targets.get(i);
      }
      System.out.printf("%f\n", prediction);
      RSQ += Math.pow(target - prediction, 2);
    }
    System.out.printf("RSQ: %f\n", RSQ);
  }
  
  public void testQuiet(ArrayList<Double[]> data, ArrayList<Double> targets) {
    RSQ = 0;
    double prediction, target = 0;
    for (int i=0; i<data.size(); i++) {
      updateNodes(data, i);
      if (NORMALIZE) {
        prediction = deNormalize(outputNode.calculate());
        target = deNormalize(targets.get(i));
      } else {
        prediction = outputNode.calculate();
        target = targets.get(i);
      }
      RSQ += Math.pow(target - prediction, 2);
    }
  }
  
  public double normalize(double x) {
    //return (x-avg)/(4*stdDev);
    return (x-avg)/NORM_DIVISOR;
  }
  public double deNormalize(double x) {
    return (x*NORM_DIVISOR)+avg;
    //return (x*4*stdDev) + avg;
  }
}
