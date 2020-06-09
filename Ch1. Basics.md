---
#### Formal Defininition
_A computer  program is said to learn from experience **E** with respect to some class of tasks **T** and performance measure **P**, if its performance at tasks in **T**, as measured by **P**, improves with experience **E**._

---

#### Tasks that could be classified as Machine Learning tasks
* **Classification or categorization**: This typically encompasses the list of problems tasks where the machine has to take in data points or samples and assign a class or category to each sample. A simple example would be classifying animal images into dogs, cats, zebras.
* **Regression**:These types of tasks usually involve performing a prediction such that a real numerical value is the output instead of a class or category for an input data point. The best way to understand a regression task would be to take the case of a real-world problem of predicting housing prices considering the plot area, number of floors, bathrooms, bedrooms, and kitchen as input attributes for each data point.
* **Anomaly detection**: These tasks involve the machine going over event logs, transaction logs, and other data points such that it can find anomalous or unusual patterns or events that are different from the normal behavior. Examples for this include trying to find denial of service attacks from logs, indications of fraud, and so on.
* **Structured annotation**: This usually involves performing some analysis on input data points and adding structured metadata as annotations to the original data that depict extra information and relationships among the data elements. Simple examples would be annotating text with their parts of speech, named entities, grammar, and sentiment. Annotations can also be done for images like assigning specific categories to image pixels, annotate specific areas of images based on their type, location, and so on.
* **Translation**: Automated machine translation tasks are typically of the nature such that if you have input data samples belonging to a specific language, you translate it into output having another desired language. Natural language based translation is definitely a huge area dealing with a lot of text data.
* **Clustering or grouping**: Clusters or groups are usually formed from input data samples by making the machine learn or observe inherent latent patterns, relationships and similarities among the input data points themselves. Usually there is a lack of pre-labeled or pre-annotated data for these tasks hence they form a part of unsupervised Machine Learning. Examples would be grouping similar products, events and entities.
* **Transcriptions**: These tasks usually entail various representations of data that are usually continuous and unstructured and converting them into more structured and discrete data elements. Examples include speech to text, optical character recognition, images to text, and so on.

---

The major fields or domains associated with ML include the following.

 - Articficial Intelligence
 - Natural Language processing
 - Data Mining
 - Mathematics
 - Statistics
 - Computer Science
 - Deep learning
 - Data Science
 
 ---

#### Some mathematical concepts

**Scalar**
* A scalar usually denotes a single number as opposed to a collection of numbers. A simple example might be x = 5 or x ∈ R, where x is the scalar element pointing to a single number or a real-valued single number.

**Vector**
* A vector is defined as a structure that holds an array of numbers which are arranged in order. This basically means the order or sequence of numbers in the collection is important. Vectors can be mathematically denoted as x = [x 1 , x 2 , ..., x n ], which basically tells us that x is a one-dimensional vector having n elements in the array. Each element can be referred to using an array index determining its position in the vector.
    ```python
    x = [1, 2, 3, 4, 5]
    import numpy as np
    x = np.array([1, 2, 3, 4, 5])
    ```

**Matrix**
* A matrix is a two-dimensional structure that basically holds numbers. It’s also often referred to as a 2D array. Each element can be referred to using a row and column index as compared to a single vector index in case of vectors.
	```python
	import numpy as np
	m = np.array([[1, 5, 2],
				 [4, 7, 4],
				 [2, 0, 9]])
	
	print(m)
	
	# View shape of the matrix
	print(m.shape) # (3, 3)
	
	# Transpose of matrix
	print('Matrix Transpose:\n', m.transpose(), '\n')
	
	# Determinant of matrix
	print('Matrix Determinant:\n', np.linalg.det(m), '\n')
	
	# Matrix inverse
	m_inv = np.linalg.inv(m)
	print('Matrix Inverse:\n', m_inv, '\n')
	```

**Tensor**
* You can think of a tensor as a generic array. Tensors are basically arrays with a variable number of axes. An element in a three-dimensional tensor T can be denoted by T x,y,z where x, y, z denote the three axes for specifying element T.

**Norm**
* The norm is a measure that is used to compute the size of a vector often also defined as the measure of distance from the origin to the point denoted by the vector. Mathematically, the pth norm of a vector is denoted as follows.

![$ L^p = ||x_p|| = \biggl(\sum_{i}|x_i|^p\biggr)^\frac{1}{p} $](https://render.githubusercontent.com/render/math?math=%24%20L%5Ep%20%3D%20%7C%7Cx_p%7C%7C%20%3D%20%5Cbiggl(%5Csum_%7Bi%7D%7Cx_i%7C%5Ep%5Cbiggr)%5E%5Cfrac%7B1%7D%7Bp%7D%20%24)

* Such that p ≥ 1 and p ∈ R. Popular norms in Machine Learning include the L 1 norm used extensively in Lasso regression models and the L 2 norm, also known as the Euclidean norm, used in ridge regression models.

**Eigen Decomposition**
* See this [article](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix).
	```python
	# Eigen decomposition
	m = np.array([[1, 5, 2],
				  [4, 7, 4],
				  [2, 0, 9]])
	
	eigen_vals, eigen_vecs = np.linalg.eig(m)
	
	print('Eigen values:', eigen_vals, '\n')
	print('Eigen vectors:\n', eigen_vecs, '\n')
	```
**Singuar Value Decomposition**
* The process of singular value decomposition, also known as SVD, is another matrix decomposition or factorization process such that we are able to break down a matrix to obtain singular vectors and singular values. Any real matrix will always be decomposed by SVD even if eigen decomposition may not be applicable in some cases.
![$M_{m\times n} = U_{m \times n}S_{m \times n}V^T_{n \times n}$](https://render.githubusercontent.com/render/math?math=%24M_%7Bm%5Ctimes%20n%7D%20%3D%20U_%7Bm%20%5Ctimes%20n%7DS_%7Bm%20%5Ctimes%20n%7DV%5ET_%7Bn%20%5Ctimes%20n%7D%24)
	```python
	# SVD
	m = np.array([[1, 5, 2],
				  [4, 7, 4], 
				  [2, 0, 9]])
	
	U, S, VT = np.linalg.svd(m)
	
	print('Getting SVD outputs:-\n')
	print('U:\n', U, '\n')
	print('S:\n', S, '\n')
	print('VT:\n', VT, '\n')
	```

**Random Variable**
* Used frequently in probability and uncertainty measurement, a random variable is basically a variable that can take on various values at random. These variables can be of discrete or continuous type in general.

**Probability Distribution**
* A probability distribution is a distribution or arrangement that depicts the likelihood of a random variable or variables to take on each of its probable states. There are usually two main types of distributions based on the variable being discrete or continuous.

**Probability Mass Function**
* A probability mass function, also known as PMF, is a probability distribution over discrete random variables. Popular examples include the Poisson and binomial distributions.

**Probability Distribution Function**
* A probability density function, also known as PDF, is a probability distribution over continuous random variables. Popular examples include the normal, uniform, and student’s T distributions.

**Conditional Probability**
* The conditional probability rule is used when we want to determine the probability that an event is going to take place, such that another event has already taken place. This is mathematically represented as follows.</br>
<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{equation}&space;\color{black}&space;P(A&space;|&space;B)&space;=&space;\frac{P(B&space;|&space;A)P(A)}{P(B)}&space;\end{equation}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{equation}&space;\color{black}&space;P(A&space;|&space;B)&space;=&space;\frac{P(B&space;|&space;A)P(A)}{P(B)}&space;\end{equation}" title="\begin{equation} \color{black} P(A | B) = \frac{P(B | A)P(A)}{P(B)} \end{equation}" /></a>

**Bayes Theorem**
* This is another rule or theorem which is useful when we know the probability of an event of interest P(A), the conditional probability for another event based on our event of interest P(B | A) and we want to determine
the conditional probability of our event of interest given the other event has taken place P(A | B). This can be defined mathematically using the following expression.
![$P(A | B) = \frac{P(B | A)P(A)}{P(B)}$](https://render.githubusercontent.com/render/math?math=%24P(A%20%7C%20B)%20%3D%20%5Cfrac%7BP(B%20%7C%20A)P(A)%7D%7BP(B)%7D%24)

**Statistics**
	```python
	# Descriptive statistics
	import scipy as sp
	import numpy as np
	
	# get data
	nums = np.random.randint(1, 20, size=(1, 15))[0]
	print('Data: ' nums)

	# get descriptive stats
	print('Mean:', sp.mean(nums))
	print('Median:', sp.median(nums))
	print('Mode:', sp.stats.mode(nums))
	print('Standard Deviation:', sp.std(nums))
	print('Variance:', sp.var(nums))
	print('Skew:', sp.stats.skew(nums))
	print('Kurtosis:', sp.stats.kurtosis(nums))
	```

---

#### Supervised Learning
Supervised learning methods or algorithms include learning algorithms that take in data samples (known as training data) and associated outputs (known  as labels or responses) with each data sample during the model training process. The main objective is to learn a mapping or association between input data samples x and their corresponding outputs y based on multiple training data instances. This learned knowledge can then be used in the future to predict an output y′ for any new input data sample x′ which was previously unknown or unseen during the model training process.

Supervised learning methods are of two major classes based on the
type of ML tasks they aim to solve.

 - **Classification**
	 - The classification based tasks are a sub-field under supervised Machine Learning, where the key objective is to predict output labels or responses that are categorical in nature for input data based on what the model has learned in the training phase. Output labels here are also known as classes or class labels are these are categorical in nature meaning they are unordered and discrete values. Thus, each output response belongs to a specific discrete class or category.
 
 - **Regression**
	 - Machine Learning tasks where the main objective is value estimation can be termed as regression tasks. Regression based methods are trained on input data samples having output responses that are continuous numeric values unlike classification, where we have discrete categories or classes. Regression models make use of input data attributes or features (also called explanatory or independent variables) and their corresponding continuous numeric output values (also called as response, dependent, or outcome variable) to learn specific relationships and associations between the inputs and their corresponding outputs. With this knowledge, it can predict output responses for new, unseen data instances similar to classification but with continuous numeric outputs.

#### Unsupervised Learning

Supervised learning methods usually require some training data where the outcomes which we are trying to predict are already available in the form of discrete labels or continuous values. However, often we do not have the liberty or advantage of having pre-labeled training data and we still want to extract useful insights or patterns from our data. In this scenario, unsupervised learning methods are extremely powerful. These methods are called unsupervised because the model or algorithm tries to learn inherent latent structures, patterns and relationships from given data without any help or supervision like providing annotations in the form of labeled outputs or outcomes.

Unsupervised learning methods can be categorized under the following
broad areas of ML tasks relevant to unsupervised learning.
- **Clustering**
	- Clustering methods are Machine Learning methods that try to find patterns of similarity and relationships among data samples in our dataset and then cluster these samples into various groups, such that each group
or cluster of data samples has some similarity, based on the inherent attributes or features. These methods are completely unsupervised because they try to cluster data by looking at the data features without any prior training, supervision, or knowledge about data attributes, associations, and relationships.

- **Dimensionality reduction**
	- Once we start extracting attributes or features from raw data samples, sometimes our feature space gets bloated up with a humongous number of features. This poses multiple challenges including analyzing and visualizing data with thousands or millions of features, which makes the feature space extremely complex posing problems with regard to training models, memory, and space constraints. In fact this is referred to as the “curse of dimensionality”. Unsupervised methods can also be used in these scenarios, where we reduce the number of features or attributes for each data sample. These methods reduce the number of feature variables by extracting or selecting a set of principal or representative features. 
	
- **Anomaly detection**
	- The process of anomaly detection is also termed as outlier detection, where we are interested in finding out occurrences of rare events or observations that typically do not occur normally based on historical
data samples. Sometimes anomalies occur infrequently and are thus rare events, and in other instances, anomalies might not be rare but might occur in very short bursts over time, thus have specific patterns. Unsupervised learning methods can be used for anomaly detection such that we train the algorithm on the training dataset having normal, non-anomalous data samples. Once it learns the necessary data representations, patterns, and relations among attributes in normal samples, for any new data sample, it
would be able to identify it as anomalous or a normal data point by using its learned knowledge.

- **Association-rule mining**
	- Typically association rule-mining is a data mining method use to examine and analyze large transactional datasets to find patterns and rules of interest. These patterns represent interesting relationships and associations, among various items across transactions. Association rule-mining is also often termed as market basket analysis, which is used to analyze customer shopping patterns. Association rules help in detecting and predicting transactional patterns based on the knowledge it gains from training transactions. Using this technique, we can answer questions like what items do people tend to buy together, thereby indicating frequent item sets. We can also associate or correlate products and items, i.e., insights like people who buy beer also tend to buy chicken wings at a pub.

- **Semi-Supervised Learning**
	- The semi-supervised learning methods typically fall between supervised and unsupervised learning methods. These methods usually use a lot of training data that’s unlabeled (forming the unsupervised learning component) and a small amount of pre-labeled and annotated data (forming the supervised learning component). Multiple techniques are available in the form of generative methods, graph based methods, and heuristic based methods.

- **Reinforcment Learning**
	- The reinforcement learning methods are a bit different from conventional supervised or unsupervised methods. In this context, we have an agent that we want to train over a period of time to interact with a specific environment and improve its performance over a period of time with regard to the type of actions it performs on the environment. Typically the agent starts with a set of strategies or policies for interacting with the environment. On observing the environment, it takes a particular action based on a rule or policy and by observing the current state of the environment. Based on the action, the agent gets a reward, which could be beneficial or detrimental in the form of a penalty. It updates its current policies and strategies if needed and this iterative process continues till it learns enough about its environment to get the desired rewards.

- **Batch Learning**
	- Batch learning methods are also popularly known as offline learning methods. These are Machine Learning methods that are used in end-to-end Machine Learning systems where the model is trained using all the
available training data in one go. Once training is done and the model completes the process of learning, on getting a satisfactory performance, it is deployed into production where it predicts outputs for new data samples. However, the model doesn’t keep learning over a period of time continuously with the new data. Once the training is complete the model stops learning. Thus, since the model trains with data in one single batch and it is usually a one-time procedure, this is known as batch or _offline learning_.

- **Online Learning**
	- Online learning methods work in a different way as compared to batch learning methods. The training data is usually fed in multiple incremental batches to the algorithm. These data batches are also known as
mini-batches in ML terminology. However, the training process does not end there unlike batch learning methods. It keeps on learning over a period of time based on new data samples which are sent to it for prediction. Basically it predicts and learns in the process with new data on the fly without have to re-run the whole model on previous data samples.

- **Instance based learning**
	- There are various ways to build Machine Learning models using methods that try to generalize based on input data. Instance based learning involves ML systems and methods that use the raw data points themselves to figure out outcomes for newer, previously unseen data samples instead of building an explicit model on training data and then testing it out.

- **Model based learning**
	- The model based learning methods are a more traditional ML approach toward generalizing based on training data. Typically an iterative process takes place where the input data is used to extract features and models are built based on various model parameters (known as hyperparameters). These hyperparameters are optimized based on various model validation techniques to select the model that generalizes best on the training data and some amount of validation and test data (split from the initial dataset). Finally, the best model is used to make predictions or decisions as and when needed.

---
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE5Nzc5MzgyMjgsMTAzODc3MjQyNCwtOT
EyNDI0MDE0XX0=
-->