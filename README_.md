# met_access
Journey through the open met database. Notes on my efforts to clean the data are documented in `development/explore.ipynb`. Source code for Flask app can be found in `src/` folder. Integration Tests can be found in `/test` folder. Templates for infrastructure can be found in `/infrastructure` folder. 

###Notes
*Exploring the commit history shows the data is updated once a week. It is not necessary to include a public `/update` endpoint for the user to refresh the source data. Ideally, I would create a job to update the source data on my instance whenever a new version is published. This can be an exercise to enhace the project. It will not be included in this version.
*The source data is ~250MB. We can store this object in memory, along with its cleaned couterparty. 
