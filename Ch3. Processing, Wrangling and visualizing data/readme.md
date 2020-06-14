#### Data Collection
Data is at the center of everything around us, which is a tremendous opportunity. Yet this also presents the fact that it must be present in different formats, shapes, and sizes. Its omnipresence also means that it exists in systems such as legacy machines (say mainframes), web (say web sites and web applications), databases, flat files, sensors, mobile devices, and so on. Let’s look at some of the most commonly occurring data formats and ways of collecting such data.
- CSV
	- A CSV data file is one of the most widely available formats of data. It is also one of the oldest formats still used and preferred by different systems across domains. Comma Separated Values (CSV) are data files that contain data with each of its attributes delimited by a “,” (a comma).
	- CSVs come in different variations and just changing the delimiter to a tab makes one a TSV (or a tab separated values) file. The basic ideology here is to use a unique symbol to delimit/separate different attributes.
		```py
		import pandas as pd
		
		data = pd.read_csv(file_name, sep=delimiter)
		```
	- With a single line and a few optional parameters (as per requirements), pandas extracts data from a CSV file into a dataframe, which is a tabular representation of the same data. One of the major advantages of using pandas is the fact that it can handle a lot of different variations in CSV files, such as files with or without headers, attribute values enclosed in quotes, inferring data types, and many more. Also, the fact that various machine learning libraries have the capability to directly work on pandas dataframes, makes it virtually a de facto standard package to handle CSV files.

- JSON
	- Java Script Object Notation (JSON) is one of the most widely used data interchange formats across the digital realm. JSON is a lightweight alternative to legacy formats like XML (we shall discuss this format next). JSON is a text format that is language independent with certain defined conventions. JSON is a human-readable format that is easy/simple to parse in most programming/scripting languages. A JSON file/object is simply a collection of name(key)-value pairs. Such key-value pair structures have corresponding data structures available in programming languages in the form of dictionaries (Python dict), struct, object, record, keyed lists, and so on.
		```py
		import pandas as pd
		
		data = pd.read_json(file_name, orient="record")
		```
	
- **Data Description**
	- **Numeric**
	This is simplest of the data types available. It is also the type that is directly usable and understood by most algorithms (though this does not imply that we use numeric data in its raw form). Numeric data represents scalar information about entities being observed, for instance, number of visits to a web site, price of a product, weight of a person, and so on. Numeric values also form the basis of vector features, where each dimension is represented by a scalar value. The scale, range, and distribution of numeric data has an implicit effect on the algorithm and/or the overall workflow. For handling numeric data, we use techniques such as normalization, binning, quantization, and many more to transform numeric data as per our requirements.

	- **Text**
	Data comprising of unstructured, alphanumeric content is one of most common data types. Textual data when representing human language content contains implicit grammatical structure and meaning. This type of data requires additional care and effort for transformation and understanding.
	
	- **Categorical**
	This data type stands in between the numeric and text. Categorical variables refer to categories of entities being observed. For instance, hair color being black, brown, blonde and red or economic status as low, medium, or high. The values may be represented as numeric or alphanumeric, which describe properties of items in consideration. Based on certain characteristics, categorical variables can be seen as: 
		- Nominal: Without any ordering.
		- Ordinal : With ordering ex. low, medium, high

* **Data Wrangling**
Data wrangling or data munging is the process of cleaning, transforming, and mapping data from one form to another to utilize it for tasks such as analytics, summarization, reporting, visualization, and so on. 
	
	- **Understanding Data**
		```py
		# df is the pandas dataframe
		
		print('Number of rows::', df.shape[0])
		print('Number of columns::', df.shape[1])
		print('Column names::', df.columns.values.tolist())
		print('Column data types::',  df.types())

		print('Columns with missing values::', df.columns[df.isnull().any()].tolist())
		print('Number of rows with missing values::', len(pd.isnull(df).any(1).nonzero()[0].tolist())
		
		print('General Stats::')
		print(df.info())
		
		print('Summary Stats::')
		print(df.describe()) 
	
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTU1MzI0NjI0OCwtNDIwNDU2MDldfQ==
-->