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
 * **Accessing  Array Elements**
	 * Basic Array indexing and slicing:
		 * Ndarray can leverage the basic indexing operations that are followed by the list class, i.e. list object [obj]. If the obj is not an ndarray object, then the indexing is said to be basic indexing.
		 * One important point to remember is that basic indexing will always return a view of the original array. It means that it will only refer to the original array and any change in values will be reflected in the original array also.
		 ```py
		 arr = np.arange(12).reshape(2, 2, 3)
		
		print(arr[0])
		```
	* Slicing:
		```py
		arr = np.arange(10)
		print(arr[5:]) # [5, 6, 7, 8, 9]
		print(arr[5:8]) # [5, 6, 7]  last indexis excluded
		print(arr[:-5]) # [0, 1, 2, 3, 4]
		```
	 * Advanced Indexing
		 * The difference in advanced indexing and basic indexing comes from the type of object being used to reference the array. If the object is an ndarray object (data type int or bool) or a non-tuple sequence object or a tuple object containing an ndarray (data type integer or bool), then the indexing being done on the array is said to be advanced indexing.
		 * **Note**: Advanced indexing will always return the copy of the original array data.
		```py
		arr = np.arange(9).reshape(3, 3)
		print(arr[[0, 1, 2], [1, 0, 0]]) # [1, 3, 6]
		```
	* Boolean Indexing
		* This advanced indexing occurs when the reference object is an array of Boolean values. This is used when we want to access data based on some conditions, in that case, Boolean indexing can be used.
		```py
		cities = ["delhi", "bangalore", "mumbai", "chennai", "bhopal"]
		city_data = np.random.randn(5,3)
		print(city_data[cities =="delhi"])
		city_data[city_data >0]
		```
* **Operations on Arrays**
	* Numpy provides a rich set of functions that we can leverage for various operations on arrays.
	* Universal functions are functions that operate on arrays in an element by element fashion. The implementation of Ufunc is vectorized, which means that the execution of Ufuncs on arrays is quite fast. The Ufuncs implemented in the numpy package are implemented in compiled C code for speed and efficiency. But it is possible to write custom functions by extending the numpy.ufunc class of the numpy package.
	```py
	arr = np.arange(15).reshape(3, 5)
	print(arr)
	print(arr + 5)
	print(arr * 2)
	```
	* Also see how broadcasting is done in NumPy.
	
* **Linear Algebra using NumPy**
	* Linear algebra is an integral part of the domain of Machine Learning. Most of the algorithms we will deal with can be concisely expressed using the operations of linear algebra. One of the most widely used operations in linear algebra is the dot product. This can be performed on two compatible (brush up on your matrices and array skills if you need to know which arrays are compatible for a dot product) ndarrays by using the dot function.
<!--stackedit_data:
eyJoaXN0b3J5IjpbMjM5MzM4ODYwXX0=
-->