package txtmine;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class TxtMiningDriver {
    public static void main(String[] args) {
        System.out.println("Starting Preprocessing");
        List<String> paths = new ArrayList<String>();
        Collections.addAll(paths,"c1article01.txt","c1article02.txt","c1article03.txt","c1article04.txt","c1article05.txt","c1article06.txt","c1article07.txt","c1article08.txt","c4article01.txt","c4article02.txt","c4article03.txt","c4article04.txt","c4article05.txt","c4article06.txt","c4article07.txt","c4article08.txt","c7article01.txt","c7article02.txt","c7article03.txt","c7article04.txt","c7article05.txt","c7article06.txt","c7article07.txt","c7article08.txt");
        
        List<List> docs = new ArrayList<List>();
        HashMap<String, Integer> ngrams = new HashMap<String, Integer>();
        HashMap<String, Integer> fcol = new HashMap<String, Integer>();
        ArrayList<Map> mapd = new ArrayList<Map>();
        Preprocessing proc = new Preprocessing();
        
        for(int i = 0; i<24; i++) {
        	docs.add(proc.process(paths.get(i)));
        	proc.ngrams(ngrams, docs.get(i));
        }
	    
        // Get the iterator over the HashMap
        Iterator<Map.Entry<String, Integer> >
            iterator = ngrams.entrySet().iterator();
        // Iterate over the HashMap
        while (iterator.hasNext()) {
            // Get the entry at this iteration
            Map.Entry<String, Integer>  entry = iterator.next();
            // Check if this value is the required value  (determines how many times n-gram needs to be found)
            if (entry.getValue() < 2) {
                // Remove this entry from HashMap
                iterator.remove();
            }
        }
        
        for(int i = 0; i<24; i++) {
        	 HashMap<String, Integer> temp = proc.adjForNgrams(ngrams, docs.get(i));
        	 mapd.add(temp);
        }
        
        
        fcol =  proc.adjFcol(mapd);
        HashMap<Integer, String> revfcol = proc.revFcol(fcol);
        
        MatrixGenerator mg = new MatrixGenerator();
        int [][] regmtx = new int [24][fcol.size()];
        for(int i = 0; i < 24; i++) {
        	mg.genreg(fcol, mapd.get(i),regmtx, i);
        }
        
        // Generated tf-idf matrix
        double [][] tfidf = mg.tfidf(regmtx, 24, fcol.size());
        
        
        // Next 30 line take the tf-idf matrix and generate top 4 words for the original 3 cluster of documents (8 documents per folder)
        HashMap<String, Double> clus1 = mg.clusTxt(tfidf, revfcol, 0, 8);
        HashMap<Double, String> clus1rev = proc.revClus(clus1);
        HashMap<String, Double> clus2 = mg.clusTxt(tfidf, revfcol, 8, 16);
        HashMap<Double, String> clus2rev = proc.revClus(clus2);
        HashMap<String, Double> clus3 = mg.clusTxt(tfidf, revfcol, 16, 24);
        HashMap<Double, String> clus3rev = proc.revClus(clus3);
        
        // Creates and orders the list of top results
        int topRes = 10;
        List<Double> list1 = new ArrayList<Double>(clus1.values());
        Collections.sort(list1, Collections.reverseOrder());
        List<Double> top5c1 = list1.subList(0, topRes);
        
        List<Double> list2 = new ArrayList<Double>(clus2.values());
        Collections.sort(list2, Collections.reverseOrder());
        List<Double> top5c2 = list2.subList(0, topRes);
        
        List<Double> list3 = new ArrayList<Double>(clus3.values());
        Collections.sort(list3, Collections.reverseOrder());
        List<Double> top5c3 = list3.subList(0, topRes);
        
        // Writes out the main topics for each folder to topics.txt
		try {
		      FileWriter myWriter = new FileWriter("topics.txt");
		      myWriter.write("Folder 1:");
		      for(int i = 1; i < topRes + 1; i++) {
		    	 Double key = top5c1.get(i-1);
		    	 myWriter.write(" " + i + "- " + clus1rev.get(key));
		      }
		      myWriter.write("\r\n");
		      
		      myWriter.write("Folder 2:");
		      for(int i = 1; i < topRes + 1; i++) {
		    	 Double key = top5c2.get(i-1);
		    	 myWriter.write(" " + i + "- " + clus2rev.get(key));
		      }
		      myWriter.write("\r\n");
		      
		      myWriter.write("Folder 3:");
		      for(int i = 1; i < topRes + 1; i++) {
		    	 Double key = top5c3.get(i-1);
		    	 myWriter.write(" " + i + "- " + clus3rev.get(key));
		      }
		      
		      myWriter.close();
		      System.out.println("Successfully wrote keywords to topics.txt.");
		    } catch (IOException e) {
		      System.out.println("An error occurred.");
		      e.printStackTrace();
		    }

	    KMeans clustering = new KMeans(3, tfidf, 10, true, .001, false, true);
	    
	    int [] assignment  = clustering.getAssignment();
	    
	    System.out.println();
	    System.out.println("----------------------------------------");
	    System.out.println("CLLUSTERING USING EUCLIDEAN DISTANCE");
	    System.out.println();
	    System.out.println("Cluster Assignment (0,1 or 2)- documents printed in ascending order (c1article01...c7article08)");
	    for (int j = 0; j < assignment.length; j++) {
    		System.out.print(assignment[j] + " ");
	    }
	    
	    int f8 = mode(assignment, 0, 8, -1);
	    int s8 = mode(assignment, 8, 16, f8);
	    int t8 = 3 - s8 - f8;
	    int [] actualAssign = {f8,f8,f8,f8,f8,f8,f8,f8,s8,s8,s8,s8,s8,s8,s8,s8,t8,t8,t8,t8,t8,t8,t8,t8};
	    
	    System.out.println();
	    System.out.println();
	    System.out.println("Actual Assignment (0,1 or 2)- documents printed in ascending order (c1article01...c7article08)");
	    for (int j = 0; j < actualAssign.length; j++) {
    		System.out.print(actualAssign[j] + " ");
	    }
	    
	    int [][] confMtx = mg.confMtx(assignment, actualAssign);
	    
	    System.out.println();
	    System.out.println();
	    System.out.println("Confusion Matrix for Folder C1, C4, C7 (Rows- Predicted Class, Columns- Actual Class )");
	    for (int i = 0; i < 3; i++) {
	    	for (int j = 0; j < 3; j++) {
	    		System.out.printf("%3d", confMtx[i][j]);
	    	}
	    	System.out.println();
	    }
	    System.out.println();
	    
	    System.out.println("Folder C1- " + mg.printAnlys(confMtx, 1));
	    System.out.println("Folder C4- " + mg.printAnlys(confMtx, 2));
	    System.out.println("Folder C7- " + mg.printAnlys(confMtx, 3));
	    System.out.println("----------------------------------------");
	    System.out.println();
	    
	    KMeans clustering2 = new KMeans(3, tfidf, 10, true, .1, true, false);
	    int [] assignment2  = clustering2.getAssignment();

	    System.out.println();
	    System.out.println("----------------------------------------");
	    System.out.println("CLLUSTERING USING COSINE SIMILARITY ");
	    System.out.println();
	    System.out.println("Cluster Assignment (0,1 or 2)- documents printed in ascending order (c1article01...c7article08)");
	    for (int j = 0; j < assignment2.length; j++) {
	    	System.out.print(assignment2[j] + " ");
	    }

	    int f82 = mode(assignment2, 0, 8, -1);
	    int s82 = mode(assignment2, 8, 16, f82);
	    int t82 = 3 - s82 - f82;
	    int [] actualAssign2 = {f82,f82,f82,f82,f82,f82,f82,f82,s82,s82,s82,s82,s82,s82,s82,s82,t82,t82,t82,t82,t82,t82,t82,t82};
	    System.out.println();
	    System.out.println();
	    System.out.println("Actual Assignment (0,1 or 2)- documents printed in ascending order (c1article01...c7article08)");
	    for (int j = 0; j < actualAssign2.length; j++) {
	    	System.out.print(actualAssign2[j] + " ");
	    }

	    int [][] confMtx2 = mg.confMtx(assignment2, actualAssign2);

	    System.out.println();
	    System.out.println();
	    System.out.println("Confusion Matrix for Folder C1, C4, C7 (Rows- Predicted Class, Columns- Actual Class )");
	    for (int i = 0; i < 3; i++) {
	    	for (int j = 0; j < 3; j++) {
	    		System.out.printf("%3d", confMtx2[i][j]);
	    	}
	    	System.out.println();
	    }
	    System.out.println();

	    System.out.println("Folder C1- " + mg.printAnlys(confMtx2, 1));
	    System.out.println("Folder C4- " + mg.printAnlys(confMtx2, 2));
	    System.out.println("Folder C7- " + mg.printAnlys(confMtx2, 3));
}
    
    public static int mode(int a[],int m, int n, int check) {
        int maxValue = 0, maxCount = 0, i, j;

        for (i = m; i < n; ++i) {
           int count = 0;
           for (j = m; j < n; ++j) {
              if (a[j] == a[i]) {
            	  if(a[j] != check)
            		  ++count;
              }
           }

           if (count > maxCount) {
              maxCount = count;
              maxValue = a[i];
           }
        }
        return maxValue;
     }
}
