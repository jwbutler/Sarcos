import java.util.ArrayList;
import java.util.Random;

public class MiddleNode {
  private ArrayList<InputNode> inputNodes;
  public ArrayList<Double> weights;
  private Random RNG;
  private int type;
  private double k;
  public final static int LINEAR = 1;
  public final static int LOGISTIC = 2;
  public final static int LOG2 = 3;
  
  public MiddleNode(ArrayList<InputNode> inputNodes, int type, double k) {
    RNG = new Random();
    this.inputNodes = inputNodes;
    this.type = type;
    this.k = k;
    weights = new ArrayList<Double>();
    for (int i=0; i<inputNodes.size(); i++) {
      double w = RNG.nextDouble()*2 - 1;
      weights.add(w);
    }
  }
  public MiddleNode(ArrayList<InputNode> inputNodes) {
    this(inputNodes, LINEAR, 0);
  }

  public double calculate() {
    double sum=0;
    for (int i=0; i<inputNodes.size(); i++) {
      sum += weights.get(i)*inputNodes.get(i).data;
    }
    return activation(sum);
  }

  /* http://en.wikipedia.org/wiki/Delta_rule */
  public void updateWeights(double target, double alpha) {
    double prediction = calculate();
    double sum = 0;
    for (int i=0; i<inputNodes.size(); i++) {
      sum += weights.get(i)*inputNodes.get(i).data;
    }
    /* Common factors of all the deltaWeights, pulled out for optimization */
    double dw = activationDerivative(sum)*alpha*(target-prediction);
    for (int i=0; i<inputNodes.size(); i++) {
      double deltaWeight = dw*inputNodes.get(i).data;
      weights.set(i, weights.get(i) + deltaWeight);
    }
  }
  
  public double activation(double x) {
    if (type==LINEAR) {
      return x;
    } else if (type==LOGISTIC) {
      return 1/(1+Math.exp(-k*x)); 
    } else if (type==LOG2) {
      return 2/(1+Math.exp(-k*x)) - 1;
    } else {
      return x;
    }
  }
  /* Courtesy of Wolfram.  Have not verified fuck calculus */
  public double activationDerivative(double x) {
    if (type==LINEAR) {
      return 1;
    /* Logistic function */
    } else if (type==LOGISTIC) {
      return 1*k*Math.exp(k*x) / Math.pow((Math.exp(k*x)+1),2);
    } else if (type==LOG2) {
      return 2*k*Math.exp(k*x) / Math.pow((Math.exp(k*x)+1),2);
    } else {
      return 1;
    }
  }
}
