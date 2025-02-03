# cbow
Continuous Bag of Words implementation and principal component analysis

This project is meant to explore embeddings visual representation and the validity of such representation. Principal Component Analysis techniques allows axis reduction from a large number (let's say over 1500) to 2 or 3. We can then plot those points and recognize patterns such as clustering. It is usefull to SEE various text fragments next to eachother on the same topic or realise that some strayed away and you can investigate why is that happening.

But is this reduction compatible with the original data? The way this is checked below is by comparing the point distances from both sets and validate that the relative distances between points in the source environment closely match the distances in the PCA environment, i.e. <code>dist(A,B,source) < dist(A,C,source)</code> then <code>dist(A,B,pca) < dist(A,C,pca) + slack</code> where <code>slack</code> is a small value to accomodate for scaling.

The embeddings can be used from the CBoW implementation by running <code>python run.py -t</code>. This will create a new CBoW model based on a trainig text (default <code>shakespeare.txt</code>) and run the included tests. Once the model is created you can run <code>python run.py -l -t</code> to load the generated model from the previous run and re-run the tests with the associated embeddings.

If you are not convinced you can bring on your own embeddings and run <code>python run.py -l -e embs_sample.txt</code> to test the solution for way more than 50 dimensions that CBoW currently has.

