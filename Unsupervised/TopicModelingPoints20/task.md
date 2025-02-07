### Topic modeling
Topic modeling involves finding topics $T$ that would well describe the documents $D$ with the dictionary $W$. 
Most topic models operate on "bag of words" data, i.e., they only consider word frequencies in documents, not their order. 
One of the simplest topic models is PLSA, which leads to the problem of stochastic matrix factorization:
$$F \approx \Phi \times \Theta$$ 
where 
 * $F_{W \times D}$ — the matrix of word distributions in documents (normalized frequencies)
 * $\Phi_{W \times T}$ — the matrix of word distributions in topics (model)
 * $\Theta_{T \times D}$ — the matrix of topic distributions in documents (the result of applying the model to the training data)

One could say that topic modeling algorithms perform soft biclustering of data:
 * _soft_, because objects belong not strictly to one cluster, but to several with different probabilities
 * _biclustering_, because the model simultaneously clusters words by topics and topics by documents
 
From a probabilistic point of view, the task of training the PLSA model is set as maximizing 
the incomplete likelihood by the parameters $\Phi$ and $\Theta$. The EM algorithm for the PLSA model involves 
repeating two steps:
 * E-step — estimation of topic distributions for each word in each document by the parameters $\Phi$ and $\Theta$ (step 6 below)
 * M-step — update of the parameters $\Phi$ and $\Theta$ based on the obtained estimates (steps 7 and 9)

There are various modifications of the iterative process that allow to reduce memory expenses. 
In this case, we will avoid storing the three-dimensional matrix $p_{tdw}$, 
immediately recalculating $\Theta$ for the current document, and accumulating counters $n_{wt}$ 
for subsequent recalculation of $\Phi$.
The pseudocode of the algorithm is written as follows:
 * Initialize $\phi_{wt}^0$ for all $w \in W$, $t \in T$ and $\theta_{td}^0$ for all $t \in T$, $d \in D$
 * Outer loop for iterations $i = 1 ... max_iter$:
 * $\quad$ $n_{wt}^i := 0$, $n_t^i := 0$ for all $w \in W$ and $t \in T$
 * $\quad$ Inner loop for documents $d \in D$
 * $\qquad$ $Z_w := \sum_{t \in T} \phi_{wt}^{i-1}\theta_{td}^{i-1}$ for all $w \in d$ $\cfrac{}{}$
 * $\qquad$ $p_{tdw} := \cfrac{ \phi_{wt}^{i-1}\theta_{td}^{i-1} }{ Z_w }$ (E-step)
 * $\qquad$ $\theta_{td}^{i} := \cfrac{ \sum_{w \in d} n_{dw} p_{tdw} }{ n_d }$ for all $t \in T$ (M-step)
 * $\qquad$ Increase $n_{wt}^i$ and $n_t^i$ by $n_{dw} p_{tdw}$ for all $w \in W$ and $t \in T$
 * $\quad \phi_{wt}^i := \cfrac{n_{wt}^i}{n_t^i}$ for all $w \in W$ and $t \in T$ (M-step)

Notations:
 * $p_{tdw}$ — the probability of topic $t$ for the word $w$ in document $d$
 * $\phi_{wt}$ — element of matrix $\Phi$, corresponding to the probability of the word $w$ in topic $t$
 * $\theta_{td}$ — element of matrix $\Theta$, corresponding to the probability of the topic $t$ in document $d$
 * $n_{wt}$ — element of a matrix of counters of assigning the word $w$ to the topic $t$ (normalizing this matrix gives the matrix $\Phi$)
 * $Z_w$ — element of a vector of auxiliary variables, corresponding to the word $w$
 * $n_t$ — vector of normalization constants for the matrix $n_{wt}$
 * $n_d$ — vector of normalization constants for the matrix $n_{dw}$
 * $n$ — the total number of words in the collection

To estimate the quality of the constructed model and control the convergence of the training process, 
perplexity is usually used:
$$\mathcal{P} = \exp\bigg(- \frac{\mathcal{L}}{n} \bigg) = 
\exp\bigg(- \cfrac{1}{n}\sum_{d \in D}\sum_{w \in d} n_{dw} \ln \big(\sum_{t \in T}\phi_{wt}\theta_{td} \big)\bigg)$$

This is a traditional measure of quality in topic modeling, 
based on the likelihood of the model $\mathcal{L}$. The number of iterations $max_{iter}$ in the learning algorithm 
should be selected sufficient for the perplexity to stop decreasing significantly. 
However, it is known that perplexity poorly reflects the interpretability of the found topics, 
so in addition to it, additional measures or expert estimates are usually used.

Recommendations for implementation:
 * When dividing by zero values, just replace the quotient with zero
 * The EM algorithm should be implemented using vector operations
To check the correctness of the implementation, you can first write a scalar version, then vectorize it, 
making sure that both implementations give the same result. 
Unvectorized algorithm can operate hundreds of times slower than vectorized, 
and its use can lead to the impossibility of performing the task
 * The iterative process should start by initializing the matrices $\Phi$ and $\Theta$ 
The initialization can be random, it’s important not to forget to normalize the columns of the matrices.
 * An inefficient implementation of perplexity can slow down the algorithm's operation several times

### Task (20 points)
Link to the Google-drive folder with data: [https://drive.google.com/drive/folders/1ItdvIUVtwAg0rHW3YmrCNfUZJPf4B1o3?usp=sharing
](https://drive.google.com/drive/folders/1ItdvIUVtwAg0rHW3YmrCNfUZJPf4B1o3?usp=sharing
)

Implement the described EM algorithm for the PLSA model and add a perplexity calculation to your implementation.
Apply your algorithm to the previously prepared data (combine the text from the name and description columns), considering the number of topics T = 5, and also:
Plot the perplexity value depending on the iteration (make sure the implementation is correct: the perplexity graph should be non-increasing).
For each topic, output the top-20 most probable words.
Take a close look at the obtained topics. Do you think they are interpretable?

### Bonus task
Consider a larger number of topics (10, 20) and several different initial approximations. 
Analyze the results and answer the following questions:
 * Can it be said that the interpretability of each topic changes as their number increases?
 * Is the algorithm stable to the initial approximation in terms of the identity of the top words in the corresponding topics?
 * Does the perplexity reflect the quality of the models obtained? What is the reason for the good/poor correspondence?
