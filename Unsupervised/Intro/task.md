## Unsupervised Learning

The result of the homework is a report. We prefer to accept reports in the format of 
IPython notebooks (ipynb-file) or pdf-files. Please try to make your report an interesting story, 
sequentially answering the questions from the tasks. 
In addition to the answers to the questions, there should also be code in the report, 
 however the less code there is, the better for everyone: 
less to check, and easier to find an error or supplement an experiment. 
The clarity of the answers to the questions, the tidiness of the report and code are assessed during the verification.

### Evaluation and penalties
Each of the tasks has a certain "cost" (indicated in the title). 
The maximum allowable score for the homework is 100 points. 
It is not possible to hand in the assignment after the specified delivery date. 
"Similar" solutions are considered plagiarism, and all involved students 
(including those who were plagiarized from) cannot get more than 0 points for it and lower their karma. 
If you have found a solution to any of the tasks in an open source, it is necessary to send a link to this source. 
Most likely you will not be the only one who found it, so to exclude suspicion of plagiarism, we need a link to the source.

## EM algorithm

### Crowdsourcing
Data labeling is one of the most labor-intensive tasks in machine learning. 
Crowdsourcing allows this task to be distributed among thousands of contributors, 
each of whom prepares a small part of the dataset 
(read more <a href="https://en.wikipedia.org/wiki/Crowdsourcing">in wikipedia</a>).

Labelers may make mistakes in labeling, in addition to this, there may be bots among them. 
If we ask only one labeler to label each object, it is highly likely that we will get no sufficiently high-quality labeling. 
Usually, several labelers label each object.

The labeling results need to be processed. 
The simplest method is *majority voting*. It consists of assigning each object the class most frequently assigned 
by the labelers to this object. This is a fairly good method, but it does not take into account various user features. 
Next, we will consider a method that allows us to estimate the probability that the labeler made a mistake.

### Dawid-Skene method (Dawid, Skene, 1979)
We have data $n_{ik}^u$ — the number of times that labeler $u \in U$ assigned class $k \in K$ to object $i \in I$ 
(possibly, the labeler saw this object several times). 
Let's denote $Y_{ik} = Indicator\\{\text{ object $i$ of class $k$}\\}$, these are our latent variables.

As parameters, we have:
* $\pi_{k\ell}^u$ — error rate, the probability that labeler $u$ assigned class $\ell$ instead of the correct class $k$
* $\rho_k$ — the probability of class $k$

Let's understand what the incomplete likelihood function in this task will be. First of all,

$$p_{\pi,\rho}(N, Y) = \prod_{i\in I}p(N_i, Y_i),$$

If $k$ is the class number of object $i$, then

$$p(N_i, Y_i)=\underbrace{p(\text{object $i$ of class $k$})}_{=\rho_k} \cdot p(N_i\mid\text{object $i$ of class $k$})$$

(the values of $Y_{it}$ are uniquely determined by the number of the true class, so on the right $Y_i$ disappears). Further, we assume that labelers act independently, therefore,
$$p(N_i\mid\text{object $i$ of class $k$}) = \prod_{u\in U}p(N_i^u\mid\text{object $i$ of class $k$}).$$

Let's figure out the quantity $p(N_i^u\mid\text{object $i$ of class $k$})$. It is responsible for 
which classes the $u$-th labeler assigned to the $i$-th object. 
We assume that the encounters of the labeler with the object are ordered in time, then
$$p(\text{$u$-th labeler assigned classes $k'_1,\ldots,k'_r$ to the $i$-th object}\mid\text{object $i$ of class $k$}) =$$

$$=\prod_{s}p(\text{at the $s$-th encounter with the $i$-th object, the $u$-th labeler assigned it to class $k'_s$}\mid\text{object $i$ of class $k$})$$

This probability can be rewritten as
$$\prod_{\ell \in K} \left( \pi_{k\ell}^u \right)^{n_{i\ell}^u},$$

and the final incomplete likelihood comes in the form

$$p_{\pi,\rho}(N, Y) = \prod_{i\in I}\prod_{k \in K} \left( \rho_k \prod_{u\in U} \prod_{\ell \in K} \left( \pi_{k\ell}^u \right)^{n_{i\ell}^u} \right)^{Y_{ik}}$$

We need to maximize it over $\pi$ and $\rho$.

**Clarification of the formula:** 
Outside the big brackets, the object and its class are fixed, the big bracket itself is raised to the power of 1 if the correct class of the object is considered, and to the power of 0 otherwise. Inside, first is recorded the probability that the object has a given class, and then — a review of all users and all classes that this user could assign. Finally, the probability that a user assigned a certain class to our object is recorded, which is raised to the power of how many times he assigned this class. For example, if a user saw an image of a kitten 5 times, while he said that it's a kitten 3 times, and a puppy two times, then the probability $\pi_{cat,cat}^u$ for this kitten will be counted 3 times, and the probability $\pi_{cat,dog}^u$ — 2 times.
