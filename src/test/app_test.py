import pytest
from mock import patch, MagicMock
import app


@patch("app.app_cache", autospec=True)
@patch("app.pandas", autospec=True)
def test_get_csv_dataframe_empty(mock_pandas, mock_cache):
	mock_pandas.read_csv.return_value = "SampleReturn"
	app.CSV_FILE_LOCATION = "Sample"
	mock_cache.get.return_value = None
	assert app._get_csv_dataframe() == "SampleReturn"


@patch("app.app_cache", autospec=True)
@patch("app.pandas", autospec=True)
def test_get_csv_dataframe_filled(mock_pandas, mock_cache):
	mock_pandas.read_csv.return_value = "SampleReturn1"
	app.CSV_FILE_LOCATION = "Sample"
	mock_cache.get.return_value = "SampleReturn2"
	assert app._get_csv_dataframe() == "SampleReturn2"


@patch("app.app_cache", autospec=True)
def test_get_backward_ref_table_filled(mock_cache):
	mock_cache.get.return_value = "Ref Table"
	assert app._get_backward_ref_table() == "Ref Table"


@patch("app.app_cache", autospec=True)
@patch("app.pickle", autospec=True)
@patch("app.open")
def test_get_backward_ref_table_empty(mock_open, mock_pickle, mock_cache):
	app.SWAP_FILE_LOCATION = "Sample"
	mock_open.return_value = MagicMock()
	mock_cache.get.return_value = None
	mock_pickle.load.return_value = "Loaded Return Value"
	assert app._get_backward_ref_table() == "Loaded Return Value"


@patch("app.app_cache", autospec=True)
def test_get_ml_model_filled(mock_cache):
	mock_cache.get.return_value = "Ml model"
	assert app._get_ml_model() == "Ml model"


@patch("app.app_cache", autospec=True)
@patch("app.pickle", autospec=True)
@patch("app.open")
def test_get_ml_model_empty(mock_open, mock_pickle, mock_cache):
	app.MODEL_FILE_LOCATION = "Sample"
	mock_open.return_value = MagicMock()
	mock_cache.get.return_value = None
	mock_pickle.load.return_value = "Loaded ML Model"
	assert app._get_ml_model() == "Loaded ML Model"

	