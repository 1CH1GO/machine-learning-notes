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
		```py
		A = np.array([[1,2,3],[4,5,6],[7,8,9]])
	    B = np.array([[9,8,7],[6,5,4],[1,2,3]])
		print(A.dot(B))
		```
	* Tranpose of a matrix
		```py
		A = np.arange(15).reshape(3,5)
		print(A)
		print(A.T)
		```
	* Oftentimes, we need to find out decomposition of a matrix into its constituents factors. This is called matrix factorization. This can be achieved by the appropriate functions. A popular matrix factorization method is SVD factorization (covered briefly in Chapter 1 concepts), which returns decomposition of a matrix into three different matrices. This can be done using linalg.svd function.
		```py
		print(np.linalg.svd(A))
		```
	* Linear algebra is often also used to solve a system of equations. Using the matrix notation of system of equations and the provided function of numpy, we can easily solve such a system of equation. Consider the system of equations:
		```
		7x + 5y -3z = 16
        3x - 5y + 2z = -8
        5x + 3y - 7z = 0
        ```
        ```py
        a = np.array([[7,5,-3], [3,-5,2],[5,3,-7]])
		b = np.array([16,-8,0])
		x = np.linalg.solve(a, b)
		print(x)
		```
	* Similarly, functions are there for finding the inverse of a matrix, eigen vectors and eigen values of a matrix, norm of a matrix, determinant of a matrix, and so on, some of which we covered in detail in Chapter 1.
	
---
#### Pandas
* Pandas is an important Python library for data manipulation, wrangling, and analysis. It functions as an intuitive and easy-to-use set of tools for performing operations on any kind of data.
* **Data Structures of Pandas**
	* Series
	* DataFrames
* **Series**
	* Series in pandas is a one-dimensional ndarray with an axis label. It means that in functionality, it is almost similar to a simple array. The values in a series will have an index that needs to be hashable. This requirement is needed when we perform manipulation and summarization on data contained in a series data structure.
* **DataFrame**
	* Dataframe is the most important and useful data structure, which is used for almost all kind of data representation and manipulation in pandas. Unlike numpy arrays (in general) a dataframe can contain heterogeneous data. Typically tabular data is represented using dataframes, which is analogous to an Excel sheet or a SQL table. This is extremely useful in representing raw datasets as well as processed feature sets in Machine Learning and Data Science. All the operations can be performed along the axes, rows, and columns, in a dataframe.

* **Data Retrieval**
	* Pandas provides numerous ways to retrieve and read in data. We can convert data from CSV files, databases, flat files, and so on into dataframes. We can also convert a list of dictionaries (Python dict) into a dataframe.
		* From List of dictionaries
		* From CSV files
		* From Databases

	 * **List of dictionaries to DataFrame**
		 * This is one of the simplest methods to create a dataframe. It is useful in scenarios where we arrive at the data we want to analyze, after performing some computations and manipulations on the raw data.
			 ```py
			 import pandas as pd
			 
			 d =  [{'city':'Delhi',"data":1000}, 
					  {'city':'Bangalore',"data":2000},
					  {'city':'Mumbai',"data":1000}]
			
			df = pd.DataFrame(d)
			print(df)
			``` 
			
	* **CSV Files to DataFrame**
		* CSV (Comma Separated Files) files are perhaps one of the most widely used ways of creating a dataframe. We can easily read in a CSV, or any delimited file (like TSV), using pandas and convert into a dataframe.
			```py
			import pandas as pd
			
			data = pd.read_csv('path_of_the_csv_file')
			print(data.head())
			```
			
	* **Databases to DataFrame**
		* The most important data source for data scientists is the existing data sources used by their organizations. Relational databases (DBs) and data warehouses are the de facto standard of data storage in almost all of the organizations. Pandas provides capabilities to connect to these databases directly, execute queries on them to extract data, and then convert the result of the query into a structured dataframe.
			```py
			server = 'xxxxxxxx' # Address of the database server
			user = 'xxxxxx'     # the username for the database server
			password = 'xxxxx'  # Password for the above user
			database = 'xxxxx'  # Database in which the table is present
			conn = pymssql.connect(server=server, user=user, password=password, database=database)
			query = "select * from some_table"
			df = pd.read_sql(query, conn)
			```
			
* **Data Operations**
	* **Values attribute**
		* Each pandas dataframe will have certain attributes. One of the important attributes is values. It is important as it allows us access to the raw values stored in the dataframe and if they all are homogenous i.e., of the same kind then we can use numpy operations on them.
			```py
			df = pd.DataFrame(np.random.randn(8, 3), columns=['A', 'B', 'C'])
			print(df)
		
			nparray = df.values
			print(nparray)
			```
			
	* **Missing data and fillna function**
		* In real-world datasets, the data is seldom clean and polished. We usually will have a lot of issues with data quality (missing values, wrong values and so on). One of the most common data quality issues is that of missing data. Pandas provides us with a convenient function that allows us to handle the missing values of a dataframe.
			```py
			df.iloc[4, 2] = np.nan
			print(df)
			
			df.fillna(0) # Fills NaN values with 0
			```
			
	* **Descriptive Statistics function**
		* Pandas provide a very useful function called `describe`. This function will calculate the most important statistics for numerical data in one go so that we dont have to use individual functions.

	* **Concatenating Data Frames** 
		* Most Data Science projects will have data from more than one data source. These data sources will mostly have data that’s related in some way to each other and the subsequent steps in data analysis will require them to be concatenated or joined. Pandas provides a rich set of functions that allow us to merge different data sources.
		* **Concatenating using `concat` method**
			```py
			# assume data_1 and data_2 are two dataframes
			data_combined = pd.concat([data_1, data_2]) # vertical concat
			data_combined = pd.concat([data_1, data_2], axis = 1) # horizontal concat 
			```

---

#### Scikit-learn
Scikit-learn is one of the most important and indispensable Python frameworks for Data Science and Machine Learning in Python. It implements a wide range of Machine Learning algorithms covering major areas of Machine Learning like classification, clustering, regression, and so on. All the mainstream Machine Learning algorithms like support vector machines, logistic regression, random forests, K-means clustering, hierarchical clustering, and many many more, are implemented efficiently in this library. Perhaps this library forms the foundation of applied and practical Machine Learning. Besides this, its easy-to-use API and code design patterns have been widely adopted across other frameworks too!

**Core APIs**

 - Dataset Representation
 - Estimators
 - Predictors
 - Transformers

**Advanced APIs**

- Meta Estimators
- Pipeline and feature unions
- Model tuning and selection

---


 
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE3OTk5MzQ5MDMsMTAyODE0MTY3Nl19
-->