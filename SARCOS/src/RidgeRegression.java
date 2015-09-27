import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

import Jama.*;
public class RidgeRegression {
  private Matrix X;
  private double[] targets;
  private int numRows, numCols;
  private final double LAMBDA = 0.00001;
  public static void main(String[] args) {
    new RidgeRegression();
  }
  public RidgeRegression() {
    readData("Sarcos_Data1_train.csv");
    double[] w = findWeights();

    readData("Sarcos_Data1_test.csv");
    for (int y=0; y<X.getRowDimension(); y++) {
      double prediction=0;  
      for (int x=0; x<X.getColumnDimension(); x++) {
         prediction += w[x]*X.get(y,x);
      }
      System.out.printf("%f\n", prediction);
    }
  }
  public void readData(String filename) {
    try {
      Scanner sc = new Scanner(new File(filename));
      String line = sc.nextLine();
      String[] toks = line.split(",");
      numRows = 0; // Don't count the header line
      numCols = toks.length-2; // Don't count id number or target
      while (sc.hasNextLine()) {
        sc.nextLine();
        numRows++;
      }
      double[][] data = new double[numRows][numCols];
      targets = new double[numRows];
      /* Restart at the first line. */
      sc.close();
      sc = new Scanner(new File(filename));
      sc.nextLine(); // skip the header line
      int row=0;
      while (sc.hasNextLine()) {
        line = sc.nextLine();
        toks = line.split(",");
        targets[row] = Double.parseDouble(toks[1]);
        for (int i=2; i<toks.length; i++) {
          data[row][i-2] = Double.parseDouble(toks[i]); 
        }
        row++;
      }
      X = new Matrix(data);
      sc.close();
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }
  }
  public double[] findWeights() {
    double[] w = new double[numCols];
    double[][] t= new double[1][numRows];
    for (int i=0; i<numRows; i++) {
      t[0][i] = targets[i];
    }
    Matrix targetsMatrix = new Matrix(t);
    Matrix m1 = X.transpose().times(X);
    Matrix m2 = Matrix.identity(numCols,  numCols).times(LAMBDA);
    Matrix m3 = (m1.plus(m2)).inverse();
    Matrix m4 = X.transpose().times(targetsMatrix.transpose());
    Matrix rtn = m3.times(m4);
    return rtn.transpose().getArray()[0];
  }
}
