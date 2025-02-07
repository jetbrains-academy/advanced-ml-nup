### Clusterization task (20 points)
In this task, you should compare [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), 
[DBSCAN](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) and 
[Birch](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html) algorithms 
using the previously prepared data.

Experiment with various ways of vectorizing text, for instance, you can:
 * Use only name/description, combine them into one text or concatenate their vector representations.
 * Use a `bag of words` representation or some of its modifications.
 * Reduce the dimensionality of the vector representation using PCA or word vector representations.

Or, do something more interesting that you can come up with! 
Select the best combination of vectorization and clustering algorithms and visualize the obtained clusters 
(for example, by using a tag cloud or propose your own method). 
Justify why you believe that the approach you have chosen to solve the task of clustering vacancies is the best.

### Bonus task
Try to enrich the vector representation obtained from the texts 
with other features from the `vacancies` table, and achieve better interpretability of clusters.
