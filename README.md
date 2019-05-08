# met_access
Journey through the open met database. Notes on my efforts to clean the data are documented in `development/explore.ipynb`. Source code for Flask app can be found in `src/` folder. Unit Tests can be found in `src/test/` Folder. Integration Tests can be found in `/test` folder. Templates for infrastructure can be found in `/infrastructure` folder. 

Notes
=====
* Exploring the commit history shows the data is updated once a week. It is not necessary to include a public `/update` endpoint for the user to refresh the source data. Ideally, I would create a job to update the source data on my instance whenever a new version is published. This can be an exercise to enhace the project. It will not be included in this version.
* The source data is ~250MB. We can store this object in memory, along with its cleaned couterparty. 
* I chose to serialize the ML model and store it on the computer. At runtime, I will grab these bytes from memory and de-serialize it. I will store that object in memory, so it doesn't need to be reloaded. Moreover, I will store a model_version as well as the model. Every execution we check the state, if the one on memory does not match, we will reload. Updating the model will not be included in this, but will be left as an exercise.
* I am also factorizing our  data, which will not be pleasant to present to an end user. For this, I am serializing the factor map and storing it on the machine. During runtime, every time the model is initialized, this will be deserialized and pulled into the cache. 
* For simplicity, I am leveraging pandas for my get requests. If I had a persistant data store, I would simply use that. Here, it is sufficient to store the dataframe in memory.
* Run the flask app `cd src && python3 app.py 'file location to store data' 'file location to store model bytes' 'file location to store reference table'`.
* To run unit tests `cd src && python3 -m pytest test/ --disable-pytest-warnings`.