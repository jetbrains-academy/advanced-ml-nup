### Task 2: aggregation (20 points)
Implement the following methods for aggregating crowdsourcing labeling results:
 * Majority voting
 * Dawid-Skene method
 
Both methods should return the probability of belonging an object to each class
(the final label is obtained by selecting the class with the highest probability estimate).

Note that the majority voting method can be implemented using a single aggregation function `pandas.crosstab`, 
while the Dawid-Skene method is based on an EM algorithm from the previous task. 
When using, it is worth considering that the EM algorithm converges to local optima, 
so it is reasonable running it from different initial approximations.

Apply these two methods to Toloka Aggregation Relevance 2 dataset, 
which is [here](course://Unsupervised/AggregationPoints10), 
and compare them.

### Bonus task
Try using the probabilities calculated by the majority voting method as the initial approximation 
of the class probabilities for each object in the David-Skene method, and first perform the M-step.
