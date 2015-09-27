import java.util.ArrayList;
import java.util.Random;

public class OutputNode {
  private ArrayList<MiddleNode> middleNodes;
  public ArrayList<Double> weights;
  private Random RNG;
  private int type;
  private final double K = 0.2;
  private final double X0 = 0;
  public final static int LINEAR = 1;
  public final static int LOGISTIC = 2;
  
  public OutputNode(ArrayList<MiddleNode> middleNodes, int type) {
    RNG = new Random();
    this.middleNodes = middleNodes;
    this.type = type;
    weights = new ArrayList<Double>();
    for (int i=0; i<middleNodes.size(); i++) {
      double w = RNG.nextDouble()*2 - 1;
      weights.add(w);
    }
  }
  public OutputNode(ArrayList<MiddleNode> middleNodes) {
    this(middleNodes, LINEAR);
  }

  public double calculate() {
    double sum=0;
    for (int i=0; i<middleNodes.size(); i++) {
      sum += weights.get(i)*middleNodes.get(i).calculate();
    }
    return activation(sum);
  }

  /* http://en.wikipedia.org/wiki/Delta_rule */
  public void updateWeights(double target, double alpha) {
    double prediction = calculate();
    double sum = 0;
    for (int i=0; i<middleNodes.size(); i++) {
      sum += weights.get(i)*middleNodes.get(i).calculate();
    }
    for (int i=0; i<middleNodes.size(); i++) {
      double deltaWeight = alpha*(target - prediction)*middleNodes.get(i).calculate()*activationDerivative(sum);
      //System.out.println(target + " " + prediction + " " + middleNodes.get(i).calculate() + " " + deltaWeight);
      weights.set(i, weights.get(i) + deltaWeight);
    }
  }
  
  public double activation(double x) {
    if (type==LINEAR) {
      return x;
    } else if (type==LOGISTIC) {
      return 1/(1+Math.exp(-K*(x-X0))); 
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
      return K*Math.exp(K*(x-X0)) / Math.pow((Math.exp(K*(x-X0))+1),2);
    } else {
      return 1;
    }
  }
  
}
