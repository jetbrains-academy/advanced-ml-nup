### Semisupervised learning (2 points)
In this task, you need to do the following:
 * Divide the objects that have labels into a training and a test sample 
(though it's not necessarily to split in a 70% to 30% ratio). Enrich the training sample with unlabeled objects.
 * Using the experience of completing previous task, take the "best" vector representation 
of vacancies and train [LabelSpreading](http://scikit-learn.org/stable/modules/generated/sklearn.semi_supervised.LabelSpreading.html) 
(select the best parameters, based on F1 score). 
 * Try running the algorithm several times, labeling different objects known 
 and also changing the proportions of the split, calculate the quality and visualize the results. 
  
 Can it be said that the algorithm strongly depends on the known initial objects? 
 Is there a class for which this is most noticeable?
 