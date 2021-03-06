# Text-Clustering-with-k-means++

This program takes in a group of text documents and clusters them based on a similarity measure. It can do this by first preprocessing the documents using the Stanford NLP Library. The program reads, preprocesses, tokenizes, and lemmatizes the text documents before converting them into a compact document-term matrix. Before the document term matrix is transformed into a TF-IDF matrix, a "sliding window" algorithm identifies frequently co-occurring phrases and compound words to improve the clustering. This program implements the k-means++ clustering algorithm with cosine similarity as the distance metric and measures the accuracy, precision, and recall of labeled text files.


Steps for running Txtmine

1) Go to Eclipse and use the import file feature
2) Import the folder into a new java project
3) Open the java file "TxtMiningDriver" and press run
4) topics.txt will update after each run with the keywords for each document
5) ClusterVisualization.png has picture of the generated clusters

![image](https://user-images.githubusercontent.com/70719859/130307005-f0180cb9-4ca5-4c87-9970-698a5285444b.png)
<img width="712" alt="ClustersVisualization" src="https://user-images.githubusercontent.com/70719859/130307022-3ff0b98f-8187-46fa-9bf0-ecaa9f8696cc.png">
