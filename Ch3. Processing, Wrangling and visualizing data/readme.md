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
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTQyMDQ1NjA5XX0=
-->