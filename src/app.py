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
	values, indices = pandas.factorize(df["Medium"])
	with open(SWAP_FILE_LOCATION, "wb+") as store_file:
		store_file.seek(0)
		pickle.dump(indices, store_file)
		store_file.truncate()
	app_cache.delete("model_indices")
	return

"""
Gets the ref table.
"""
def _get_backward_ref_table():
	curr_model_indices = app_cache.get("model_indices")
	if curr_model_indices:
		return curr_model_indices
	with open(SWAP_FILE_LOCATION, "rb") as store_file:
		store_file.seek(0)
		curr_model_indices = pickle.load(store_file)
	return curr_model_indices 

def _persist_ml_model(model):
	with open(MODEL_FILE_LOCATION, "wb+") as ml_file:
		ml_file.seek(0)
		pickle.dump(model, ml_file)
		ml_file.truncate()
	with open(MODEL_VERSION_FILE_LOCATION, "wb+") as ml_version_file:
		_increment_version(ml_version_file)
	return


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
	print("Stored Version: {0}\nCurrent Version: {1}\nModel: {2}".format(stored_version, curr_version, curr_model))
	if curr_version is None or curr_model is None:
		return False
	if curr_version != stored_version:
		return False
	return True


"""
Verify that the necessary params are given for prediction engine.
"""
def _verify_params(**kwargs):
	valid = False
	try:
		_dataframe = pandas.Dataframe.from_dict(kwargs)
		return True
	except Exception as e:
		print(e)
		return False
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




@app.route("/art/prediction", methods=["POST", "GET"])
def predict():
	# params = request.get_json(silent=True)
	valid = _verify_ml_model()
	return json.dumps({"Valid": valid})



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
		return NoDataFoundException(id)
	except Exception as e:
		return GeneralException(e)





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


