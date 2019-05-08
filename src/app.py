import sys
import os
import logging
import threading
import pickle
import math
import json
from config import CONFIG
from exception.exceptions import NoDataFoundException, GeneralException

from flask import Flask, request
from werkzeug.contrib.cache import SimpleCache
import numpy
import pandas
import sklearn.neighbors as nb
import requests


app = Flask(__name__)
app_cache = SimpleCache()
DATAFRAME = None



"""
Creates the dataframe and stores it in the cache. If the dataframe has been created before, fetches object from cache.
"""
def _get_csv_dataframe():
	df = app_cache.get("csv_file")
	if df is None:
		df = pandas.read_csv(CSV_FILE_LOCATION)
		app_cache.set("csv_file", df)
	return df


"""
Stores the given medium factorization map on the machine for prediction engine.
"""
def _store_backward_ref_table(df):
	ref_table = dict()
	for column in df.columns:
		ref_table[column] = dict()
		values, indices = pandas.factorize(df[column])
		for count, val in enumerate(indices):
			ref_table[column][val] = count

	with open(SWAP_FILE_LOCATION, "wb+") as store_file:
		store_file.seek(0)
		pickle.dump(ref_table, store_file)
		store_file.truncate()
	app_cache.delete("ref_table")
	return

"""
Gets the ref table.
"""
def _get_backward_ref_table():
	ref_table = app_cache.get("ref_table")
	if ref_table:
		return ref_table
	with open(SWAP_FILE_LOCATION, "rb") as store_file:
		store_file.seek(0)
		ref_table = pickle.load(store_file)
		
	app_cache.set("ref_table", ref_table)
	return ref_table 

"""
Serializes and stores the model on the computer. Increments model version.
"""
def _persist_ml_model(model):
	with open(MODEL_FILE_LOCATION, "wb+") as ml_file:
		ml_file.seek(0)
		pickle.dump(model, ml_file)
		ml_file.truncate()
	with open(MODEL_VERSION_FILE_LOCATION, "wb+") as ml_version_file:
		_increment_version(ml_version_file)
	return

"""
Gets our current ML Model.
"""
def _get_ml_model():
	_curr_model = app_cache.get("curr_ml_model")
	if _curr_model:
		return _curr_model

	with open(MODEL_FILE_LOCATION, "rb+") as ml_file:
		ml_file.seek(0)
		_curr_model = pickle.load(ml_file)
	app_cache.set("curr_ml_model", _curr_model)
	return _curr_model




def _increment_version(fp):
	fp.seek(0)
	_bytes = fp.read()
	_str = _bytes.decode("utf-8")
	curr_version = 0
	if _str != "-":
		try:
			curr_version = int(_str) + 1
		except Exception as e:
			pass
	fp.seek(0)
	app_cache.set("curr_ml_version", curr_version)
	fp.write(str(curr_version).encode("utf-8"))
	fp.truncate()


"""
Verify that our model has been initialized and trained. 
"""
def _verify_ml_model():
	stored_version = None
	with open(MODEL_VERSION_FILE_LOCATION, "r") as version_file:
		version_file.seek(0)
		_bytes = version_file.read()
		if _bytes == "-":
			return False
		else:
			stored_version = int(_bytes)
	curr_version = app_cache.get("curr_ml_version")
	curr_model = app_cache.get("curr_ml_model")
	if curr_version is None or curr_model is None:
		return False
	if curr_version != stored_version:
		return False
	return True


"""
Verify that the necessary params are given for prediction engine.
"""
def _verify_params(params):
	valid = False
	if params is None:
		return valid
	try:
		_ref_table = _get_backward_ref_table()
		for key in params.keys():
			if key not in _ref_table:
				return False
		valid = True
	except Exception as e:
		print(e)
		valid = False
	return valid


"""
Run at initialization. 
"""
def initialize_files():
	try:
		print("Requesting data from Github...")
		with requests.get(CONFIG['http_data_file'], stream=True) as response:
			response.raise_for_status()
			print("Streaming data into file...")
			with open(CSV_FILE_LOCATION, "wb+") as file:
				for chunk in response.iter_content(chunk_size=(10*1024)):
					if chunk:
						file.write(chunk)
		with open(MODEL_VERSION_FILE_LOCATION, "wb+") as version_file:
			print("Rewriting ml model...")
			version_file.seek(0)
			version_file.write("-".encode("utf-8"))
			version_file.truncate()
		return True
	except requests.exceptions.HTTPError as http_err:
		print(http_err)
	except Exception as exc:
		print(exc)

	return False



"""
Initializes the machine learning model. Trains it over 1/3 of our data. 
Persists the model on the machine and sets version to 0.
Stores backward reference table (swap table) on the machine.
"""
def train_ml_model():
	_success = False
	try:
		df = _get_csv_dataframe()
		df.dropna(subset=["Medium"], inplace=True)
		counts = df["Medium"].value_counts()
		indices = counts[counts > 1].index
		df = df[df["Medium"].isin(indices)]
		df = df.drop(["Link Resource", "River", "State"], axis=1)
		print("Storing Reference Data...")
		_store_backward_ref_table(df)
		print("Stored Reference Data Successfully...")
		prepared_dataframe = df.apply(lambda x: pandas.factorize(x)[0])
		N = len(prepared_dataframe.index)
		training_dataframe = prepared_dataframe.sample(int(math.floor(N * CONFIG['SAMPLING_SIZE'])))
		X = training_dataframe.drop(['Medium'], axis=1)
		Y = training_dataframe["Medium"]
		model = nb.KNeighborsClassifier()
		print("Training Model on Training Data...")
		model.fit(X, Y)
		print("Trained Model Successfully. Storing...")
		_persist_ml_model(model)
		print("Stored Model Successfully...")
		app_cache.set("curr_ml_model", model)
		_success = True
	except Exception as e:
		print(e)
	return _success


def _transform_raw_prediction(prediction, table):
	value = prediction[0]
	for k, v in table.items():
		if value == v:
			return k
	return "Unspecified Value"


"""
Given a set of parameters, returns a prediction from our model.
"""
@app.route("/art/prediction", methods=["POST"])
def predict():
	params = request.get_json(silent=True)
	_valid_params = _verify_params(params)
	if _valid_params:
		valid = _verify_ml_model()
		_ref_table = _get_backward_ref_table()
		if valid:
			_predict_dataframe = pandas.DataFrame()
			for column in params.keys():
				if params[column] in _ref_table[column]:
					_predict_dataframe[column] = pandas.Series([ _ref_table[column][params[column]] ])
				else:
					_predict_dataframe[column] = pandas.Series([-1])
			_keys = set(params.keys())
			_keys.add("Medium") #Add Medium as that is dropped.
			missing_column_set = set(_ref_table.keys()) - _keys
			for col in list(missing_column_set):
				_predict_dataframe[col] = pandas.Series([-1])
			if _predict_dataframe.empty:
				return json.dumps({"error": "Parameters did not contain valid data. Please check values and try again."})
			_model = _get_ml_model()
			_raw_prediction = _model.predict(_predict_dataframe)
			_transformed_prediction = _transform_raw_prediction(_raw_prediction, _ref_table["Medium"])
			return json.dumps({"Prediction": "{0}".format(_transformed_prediction)})
		else:
			return json.dumps({"error": "Machine Learning Model Not Loaded. Please wait and Try Again."})
	else:
		return json.dumps({"error": "Invalid Parameters."})
	return 



"""
Gets all information for a given id. The unique identifier in the dataframe.
"""
@app.route("/art/<id>")
def get_single_id(id):
	df = _get_csv_dataframe()
	try:
		index = int(id)
		row = df.loc[index]
		return row.to_json()
	except KeyError as ke:
		return json.dumps({"error": "No data found for ID: {0}".format(id)})
	except Exception as e:
		return json.dumps({"error": "A General Exception occurred...Stack Trace: {0}".format(str(e))})

"""
Checks that the given params are valid for a get request.
"""
def _verify_get_params(params):
	_valid_keys = {"page_number", "page_size"}
	_valid = True
	for item in params.keys():
		if item not in _valid_keys:
			_valid = False
		if not params[item].isdigit():
			_valid = False
	return _valid

"""
Returns the bottom and top index of a valid request.
"""
def _get_dataframe_indices(args, max_index):
	_page_number, _page_size = 0, CONFIG["DEFAULT_PAGE_SIZE"]
	_bottom_index = 0
	_top_index = _bottom_index + _page_size
	if "page_size" in args:
		_page_size = min(int(args["page_size"]), CONFIG["MAXIMUM_PAGE_SIZE"])
	if "page_number" in args:
		_page_number = int(args["page_number"])
	if (_page_number+1)*_page_size >= max_index:
		_top_index = max_index
		_bottom_index = _top_index - _page_size
	else:
		_bottom_index = _page_size*_page_number
		_top_index = _bottom_index + _page_size
	return _bottom_index, _top_index


"""
Gets a page of data. If no page value is given, defaults to one. If no page_size is given, defaults to default.
Info is given as URL k/v pairs.
"""
@app.route("/art")
def get_art():
	df = _get_csv_dataframe()
	args = request.args
	n = df.index.size
	_valid = _verify_get_params(args)
	if _valid:
		_bottom_index, _top_index = _get_dataframe_indices(args, n)
		try:
			_dataframe = df.loc[_bottom_index: _top_index]
			return _dataframe.to_json()
		except Exception as e:
			return json.dumps({"error": "A General Exception occurred...Stack Trace: {0}".format(str(e))})	
	else:
		return json.dumps({"error": "A General Exception occurred...Stack Trace: {0}".format(str(e))})


if __name__ == '__main__':
	global CSV_FILE_LOCATION
	global MODEL_FILE_LOCATION
	global MODEL_VERSION_FILE_LOCATION
	global SWAP_FILE_LOCATION
	CSV_FILE_LOCATION = sys.argv[1]
	MODEL_FILE_LOCATION = sys.argv[2]
	SWAP_FILE_LOCATION = sys.argv[3]
	MODEL_VERSION_FILE_LOCATION = CONFIG['version_file']
	initialized_successful = initialize_files()
	if initialized_successful:
		trained_successful = train_ml_model()

	if initialized_successful:
		print("File was initialized successfully. Booting App...")
		_get_csv_dataframe()
		app.run()


