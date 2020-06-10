#### Setting up Anaconda for Machine Learning Development

Read [this.](https://machinelearningmastery.com/setup-python-environment-machine-learning-deep-learning-anaconda/)

#### Installing libraries
`pip install required_package`

---

#### Jupyter Notebook
Read [this.](https://realpython.com/jupyter-notebook-introduction/)
- Note: If you already have anaconda installed on your pc you need not do the installation part in the above tutorial.

---

#### Numpy 
* Numpy is the backbone of Machine Learning in Python. It is one of the most important libraries in Python for numerical computations. It adds support to core Python for multi-dimensional arrays (and matrices) and fast vectorized operations on these arrays.
* **Numpy ndarray**
	* All of the numeric functionality of numpy is orchestrated by two important constituents of the numpy package, ndarray and Ufuncs (Universal function). Numpy ndarray is a multi-dimensional array object which is the core data container for all of the numpy operations. Universal functions are the functions which operate on ndarrays in an element by element fashion.
		```python
		import numpy as np
		arr = np.array([1, 2, 3, 5, 6])
		
		print(arr) # [1 2 3 5 6]
		print(arr.shape) # (5, ) 
		print(arr.dtype) # dtype('int32') dtype => datatype
		```
		
	* One important thing to keep in mind is that all the elements in an array must have the same data type. If you try to initialize an array in which the elements are mixed, i.e. you mix some strings with the numbers then all of the elements will get converted into a string type and we won’t be able to perform most of the numpy operations on that array. So a simple rule of thumb is dealing only with numeric data.
		```py
		arr = np.array([1, 'st', 'er', 2])
		
		print(arr.dtype) # dtype('<U21')
		print(np.sum(arr)) # Error!!
		```
* **Creating Arrays** 
	* Arrays can be created in multiple ways in numpy. One of the ways was demonstrated earlier to create a single-dimensional array. Similarly we can stack up multiple lists to create a multidimensional array.
		```py
		arr = np.array([[[1,2,3],[2,4,6],[8,8,8]])
		print(arr.shape) # (3, 3)
		print(arr)
		```
	* `np.zeros`: Creates a matrix of specified dimensions containing only zeroes
		```py
		arr = np.zeros((2, 4))
		print(arr)
		```
	* `np.ones`: Createa a matrix of specified dimensions containing only ones.
		```py
		arr = np.ones((2, 4))
		print(arr)
		```
	* `np.identity`: Creates an indentity matrix of specified dimensions
		```py
		arr = np.identity(3)
		print(arr)
		```
	* Often, an important requirement is to initialize an array of a specified dimension with random values. This can be done easily by using the randn function from the numpy.random package:
		```py
		arr = np.random.randn(3, 4)
		print(arr)
		```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTI2NDUzNDYyOF19
-->
