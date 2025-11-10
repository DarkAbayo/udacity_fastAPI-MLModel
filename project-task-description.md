# Project Description: Deploying a Machine Learning Model with FastAPI

## Project Instructions

### Working Environment
- Working in a command line environment is recommended for ease of use with Git and DVC. On Windows, WSL1 or 2 is recommended.

### Repositories
- Create a directory for the project and initialize Git.
  - As you work on the code, continually commit changes. Trained models you want to keep must be committed to GitHub.
- Connect your local Git repository to GitHub.

### GitHub Actions
- Set up GitHub Actions in your repository. You can use one of the pre-made GitHub Actions that runs `pytest` and `flake8` on every push and requires both to pass without error.
  - Make sure the GitHub Action uses the same Python version you used in development.

### Data
- Download `census.csv` from the data folder in the starter repository.
  - Information about the dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/census+income).
- This data is messy, try to open it in pandas and see what you get.
- To clean it, use your favorite text editor to remove all spaces.

### Model
- Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
- Write unit tests for at least 3 functions in the model code.
- Write a function that outputs the performance of the model on slices of the data.
  - Suggestion: For simplicity, the function can just output the performance on slices of just the categorical features.
- Write a model card using the provided template.

### API Creation
- Create a RESTful API using FastAPI that implements:
  - GET on the root giving a welcome message.
  - POST that does model inference.
  - Type hinting must be used.
  - Use a Pydantic model to ingest the body from POST. This model should contain an example.
    - Hint: The data has names with hyphens, and Python does not allow those as variable names. Do not modify the column names in the CSV and instead use the functionality of FastAPI/Pydantic/etc. to deal with this.
- Write 3 unit tests to test the API (one for GET and two for POST, one for each prediction).
- **Run a sanity check for your test cases:**
  - Run `python sanitycheck.py`. This script is located in the starter directory in the starter code.
  - The script scans the written test cases for the `GET()` and `POST()` APIs and generates a report.
  - The report lists all issues it recognizes. Fix the issues and run the `sanitycheck.py` script again.
  - The script uses heuristics to recognize common problems and may sometimes miss a problem or trigger a false alarm. You should still check your implementation against the project rubric to ensure your submission meets the requirements.

### API Deployment on a Cloud Application Platform
- Create an account on a cloud application platform, such as `Heroku` or `Render`. For the following steps, you can either use the web GUI or download the CLI.
  - *Note: As of November 28, 2022, Heroku has announced the removal of the free account, which is being replaced by a low-cost subscription plan. (Reference: [Removal of Heroku Free Product Plans FAQ](https://help.heroku.com/RSBRUH58/removal-of-heroku-free-product-plans-faq))*
- Create a new app and have it deployed from your GitHub repository.
  - Enable automatic deployments that only deploy if your continuous integration passes.
  - Hint: Think about how paths will differ in your local environment vs. on the cloud platform.
  - Hint: Development in Python is fast! But how fast you can iterate slows down if you rely on your CI/CD to fail before fixing an issue. I recommend running `flake8` locally before committing changes.
- Write a script that uses the `requests` module to do one POST on your live API.

## Project Rubric

### Git
- **Criterion:** Set up Git with GitHub Actions.
  - **Passing:**
    - GitHub Action should run `pytest` and `flake8` on every push to `main/master`.
    - `PyTest` must pass (by the time the project is complete, there should be at least six tests) and `flake8` must pass without errors.
    - Add either a link to your GitHub repository or a screenshot of the successful CI named `continuous_integration.png`.

### Model Building
- **Criterion:** Create a Machine Learning Model.
  - **Passing:**
    - The model should be trained on the provided data. The data should either be split to have a train-test split, or cross-validation should be performed on the entire dataset.
    - Implement all stubbed functions in the starter code or create equivalents. At minimum, there should be functions to:
      - train, save, and load the model as well as all categorical encoders.
      - perform model inference.
      - determine classification metrics.
    - Write a script that reads in the data, processes it, trains the model, and saves it along with the encoder. This script must use the functions you have written.

- **Criterion:** Write Unit Tests.
  - **Passing:**
    - Write at least 3 unit tests. Unit tests for ML can be difficult because they are stochastic – test at least whether ML functions return the expected type.

- **Criterion:** Write a function that computes model metrics on data slices.
  - **Passing:**
    - Write a function that computes performance on model slices. I.e., a function that computes performance metrics when the value of a given feature is fixed. E.g., for education, it would output the model metrics for each slice of data that has a particular value for education. You should have one set of outputs for each unique value in education.
    - Complete the stubbed function or write a new one that, for a given categorical variable, computes the metrics when its value is fixed.
    - Write a script that runs this function (or add it as part of the training script) that iterates through the different values of one of the features and outputs the model metrics for each value.
    - Output the output in a file named `slice_output.txt`.

- **Criterion:** Write a Model Card.
  - **Passing:**
    - The model card should address every section of the template. **Please use the provided [Model Card Template](https://github.com/udacity/nd0821-c3-starter-code/blob/master/starter/model_card_template.md).**
    - The model card should be written in complete sentences and include metrics on model performance. Please add both the metrics used and the performance of your model on those metrics.

### API Creation
- **Criterion:** Create a REST API.
  - **Passing:**
    - The API must implement GET and POST. GET must be on the root domain and output a greeting, and POST on another path that performs model inference.
    - Use Python type hints so that FastAPI creates automatic documentation.
    - Use a Pydantic model to process the POST body. This should implement an example (Hint: Pydantic/FastAPI offers several ways to do this, see the documentation: https://fastapi.tiangolo.com/tutorial/schema-extra-example/).
    - Add a screenshot of the documentation showing the example and name it `example.png`.

- **Criterion:** Create Tests for an API.
  - **Passing:**
    - You should write at least three test cases –
      - One test case for the GET method. This MUST test both the status code and the content of the request object.
      - One test case for EACH of the possible inferences (results/outputs) of the ML model.

### API Deployment
- **Criterion:** Deploy an app on a cloud application platform.
  - **Passing:**
    - Deployment via a GitHub repository with continuous deployment enabled. Add a screenshot showing that CD is enabled and name it `continuous_deployment.png`.
    - Add a screenshot of your browser showing the content of the GET you implemented on the root domain. Name this screenshot `live_get.png`.

- **Criterion:** Query the Live API.
  - **Passing:**
    - Write a script that sends POST to the API using the `requests` module and returns both the result of the model inference and the status code. Add a screenshot of the result. Name this `live_post.png`.
