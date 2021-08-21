package txtmine;

import java.io.IOException;
import java.util.HashSet;
import java.util.Random;

public class KMeans {

   int k;
   double[][] points;
   int iterations;
   boolean plus;
   double stopMeas;
   boolean useStop;
   boolean disMeas;
   int m;
   int n;
   double[][] centroids;
   int[] assignment;
   double WCSS;

   public KMeans(int K, double[][] Points, int Iterations, boolean Plus, double StopMeas, boolean UseStop, boolean DisMeas) {
      k = K;
      points = Points;
      iterations = Iterations;
      plus = Plus;
      stopMeas = StopMeas;
      useStop = UseStop;
      disMeas = DisMeas;
      m = Points.length;
      n = Points[0].length;
      run();
   }

   private void run() {
      double bestWCSS = Double.POSITIVE_INFINITY;
      double[][] bestCentroids = new double[0][0];
      int[] bestAssignment = new int[0];
      for (int n = 0; n < iterations; n++) {
         cluster();

         if (WCSS < bestWCSS) {
            bestWCSS = WCSS;
            bestCentroids = centroids;
            bestAssignment = assignment;
         }
      }

      WCSS = bestWCSS;
      centroids = bestCentroids;
      assignment = bestAssignment;
   }

   private void cluster() {
      chooseInitialCentroids();
      WCSS = Double.POSITIVE_INFINITY; 
      double prevWCSS;
      do {  
         assignmentStep();

         updateStep();

         prevWCSS = WCSS;
         calcWCSS();
      } while (!stop(prevWCSS));
   }

   private void assignmentStep() {
      assignment = new int[m];

      double tempDist;
      double minValue;
      int minLocation;

      for (int i = 0; i < m; i++) {
         minLocation = 0;
         minValue = Double.POSITIVE_INFINITY;
         for (int j = 0; j < k; j++) {
            tempDist = distance(points[i], centroids[j]);
            if (tempDist < minValue) {
               minValue = tempDist;
               minLocation = j;
            }
         }

         assignment[i] = minLocation;
      }

   }

   private void updateStep() {
      for (int i = 0; i < k; i++) {
         for (int j = 0; j < n; j++) {
            centroids[i][j] = 0;
         }
      }
      
      int[] clustSize = new int[k];

      for (int i = 0; i < m; i++) {
         clustSize[assignment[i]]++;
         for (int j = 0; j < n; j++) {
            centroids[assignment[i]][j] += points[i][j];
         }
      }
      
      HashSet<Integer> emptyCentroids = new HashSet<Integer>();

      for (int i = 0; i < k; i++) {
         if (clustSize[i] == 0) {
        	 emptyCentroids.add(i);
         }

         else {
            for (int j = 0; j < n; j++) {
               centroids[i][j] /= clustSize[i];
            }
         }
      }
      
      if (emptyCentroids.size() != 0) {
         HashSet<double[]> nonemptyCentroids = new HashSet<double[]>(k - emptyCentroids.size());
         for (int i = 0; i < k; i++) {
            if (!emptyCentroids.contains(i)) {
               nonemptyCentroids.add(centroids[i]);
            }
         }
         
         Random r = new Random();
         for (int i : emptyCentroids) {
            while (true) {
               int rand = r.nextInt(points.length);
               if (!nonemptyCentroids.contains(points[rand])) {
                  nonemptyCentroids.add(points[rand]);
                  centroids[i] = points[rand];
                  break;
               }
            }
         }

      }
      
   }

   private void chooseInitialCentroids() {
      if (plus)
         plusplus();
      else
         basicRandSample();
   }

   private void basicRandSample() {
      centroids = new double[k][n];
      double[][] copy = points;

      Random gen = new Random();

      int rand;
      for (int i = 0; i < k; i++) {
         rand = gen.nextInt(m - i);
         for (int j = 0; j < n; j++) {
            centroids[i][j] = copy[rand][j];
            copy[rand][j] = copy[m - 1 - i][j];
         }
      }
   }

   private void plusplus() {
      centroids = new double[k][n];       
      double[] distToClosestCentroid = new double[m];
      double[] weightedDistribution  = new double[m];

      Random gen = new Random();
      int choose = 0;

      for (int c = 0; c < k; c++) {

         if (c == 0) {
        	 choose = gen.nextInt(m);
         }

         else {
            for (int p = 0; p < m; p++) {
               double tempDistance = Distance.D2(points[p], centroids[c - 1]);
               if (c == 1) {
                  distToClosestCentroid[p] = tempDistance;
               }

               else {
                  if (tempDistance < distToClosestCentroid[p]) {
                     distToClosestCentroid[p] = tempDistance;
                  }
               }
               if (p == 0) {
                  weightedDistribution[0] = distToClosestCentroid[0];
               }
               else {
            	   weightedDistribution[p] = weightedDistribution[p-1] + distToClosestCentroid[p];
               }

            }

            double rand = gen.nextDouble();
            for (int j = m - 1; j > 0; j--) {
               if (rand > weightedDistribution[j - 1] / weightedDistribution[m - 1]) { 
                  choose = j;
                  break;
               }
               else
                  choose = 0;
            }
         }
         
         for (int i = 0; i < n; i++) {
            centroids[c][i] = points[choose][i];
         }
      }   
   }

   public boolean stop(double prevWCSS) {
      if (useStop)
         return stopTest(prevWCSS);
      else
         return prevWCSS == WCSS;
   }

   public boolean stopTest(double prevWCSS) {
      return stopMeas > 1 - (WCSS / prevWCSS);
   }

   public double distance(double[] x, double[] y) {
      return disMeas ? Distance.D1(x, y) : Distance.D3(x, y);
   }
   
   public static class Distance {
	   
      public static double D1(double[] x, double[] y) {
         if (x.length != y.length) throw new IllegalArgumentException("dimension error");
         double dist = 0;
         for (int i = 0; i < x.length; i++) 
            dist += Math.abs(x[i] - y[i]);
         return dist;
      }
      
      public static double D2(double[] x, double[] y) {
         if (x.length != y.length) throw new IllegalArgumentException("dimension error");
         double dist = 0;
         for (int i = 0; i < x.length; i++)
            dist += Math.abs((x[i] - y[i]) * (x[i] - y[i]));
         return dist;
      }
      
      public static double D3(double[] x, double[] y) {
    	  if (x.length != y.length) throw new IllegalArgumentException("dimension error");
          double sumProduct = 0;
          double sumASq = 0;
          double sumBSq = 0;
          for (int i = 0; i < x.length; i++) {
        	  sumProduct += x[i]*y[i];
        	  sumASq += x[i] * x[i];
        	  sumBSq += y[i] * y[i];
          }
          if (sumASq == 0 && sumBSq == 0) {
      		return 2.0;
          }
          return sumProduct / (Math.sqrt(sumASq) * Math.sqrt(sumBSq));
      }
   }
   
   public void calcWCSS() {
      double WCSS = 0;
      int assignedClust;

      for (int i = 0; i < m; i++) {
         assignedClust = assignment[i];
         WCSS += distance(points[i], centroids[assignedClust]);
      }     

      this.WCSS = WCSS;
   }

   public int[] getAssignment() {
      return assignment;
   }

   public double[][] getCentroids() {
      return centroids;
   }

   public double getWCSS() {
      return WCSS;
   }
}