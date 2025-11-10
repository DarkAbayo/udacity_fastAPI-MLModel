# FastAPI ML Deployment Project

This project implements a machine learning model for Census income prediction and deploys it as a RESTful API using FastAPI on Render.

## Project Structure

```
project/
├── starter/                # ML model package
│   ├── ml/
│   │   ├── data.py         # Data preprocessing functions
│   │   └── model.py        # Model training and inference functions
│   └── train_model.py      # Training script entry point
├── data/
│   └── census.csv          # Census dataset (needs cleaning)
├── main.py                 # FastAPI application (to be implemented)
├── docs/                   # German documentation (detailed, beginner-friendly)
├── requirements.txt        # Python dependencies
├── setup.py                # Package setup configuration
├── sanitycheck.py          # Test validation script
├── model_card_template.md  # Template for model documentation
└── project-task-description.md  # Detailed project requirements and rubric
```

## Environment Setup

### Using pip and venv (Recommended)
```bash
# Ensure Python 3.13 is installed
python3.13 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Development Tasks

### 1. Data Preparation
- [ ] Download `census.csv` from the data folder
- [ ] Clean the census.csv file (remove spaces using text editor)
- [ ] Verify data can be opened in pandas
- [ ] Set up DVC for data versioning (optional)
- [ ] Commit cleaned data to DVC (optional)

### 2. Model Development
- [ ] Complete `train_model()` function in `starter/ml/model.py`
  - Train model on clean data
  - Save model and encoders
- [ ] Complete `inference()` function in `starter/ml/model.py`
- [ ] Write training script (`starter/train_model.py`)
  - Load and process data
  - Train model using implemented functions
  - Save model and encoder
- [ ] Write unit tests for at least 3 model functions
  - Test that functions return expected types
  - Handle stochastic nature of ML tests
- [ ] Implement function for performance on data slices
  - Compute metrics for each unique value of categorical features
  - Output results to `slice_output.txt`
- [ ] Create model card using the provided template
  - Address every section of the template
  - Include metrics and model performance
  - Write in complete sentences

### 3. API Development
- [ ] Implement GET endpoint (welcome message) in `main.py`
  - Must be on root domain
- [ ] Implement POST endpoint (model inference) in `main.py`
  - Use different path from GET
- [ ] Create Pydantic model for POST body with example
  - Handle column names with hyphens using Pydantic Field aliases
  - See: https://fastapi.tiangolo.com/tutorial/schema-extra-example/
- [ ] Use Python type hints for automatic FastAPI documentation
- [ ] Write 3 unit tests (1 GET, 2 POST)
  - GET test: Must test both status code and response content
  - POST tests: One for each possible model prediction outcome
- [ ] Run sanity check: `python sanitycheck.py`
  - Fix any issues reported
  - Re-run until passing
- [ ] Take screenshot of API documentation showing example (`example.png`)

### 4. Git & CI/CD Setup
- [ ] Set up GitHub repository
- [ ] Configure GitHub Actions
  - Run `pytest` and `flake8` on every push to `main/master`
  - Use Python 3.13 (same as development)
  - Both must pass without errors
  - At least 6 tests should exist by project completion
- [ ] Add screenshot of successful CI (`continuous_integration.png`) or link to repository

### 5. Deployment
- [ ] Deploy to Render
  - Connect GitHub repository
  - Enable automatic deployments (only if CI passes)
  - Configure build and start commands
- [ ] Configure environment variables on Render (if needed)
- [ ] Take screenshot showing continuous deployment enabled (`continuous_deployment.png`)
- [ ] Take screenshot of browser showing GET endpoint (`live_get.png`)
- [ ] Write script to test live API using `requests` module
  - Send POST request
  - Return model inference result and status code
- [ ] Take screenshot of POST result (`live_post.png`)

## Running the Application

### Local Development
```bash
# Activate virtual environment
source .venv/bin/activate

# Run the FastAPI application
uvicorn main:app --reload
```

### Testing
```bash
# Run all tests
pytest

# Run linting
flake8

# Run sanity check
python sanitycheck.py
```

## Project Requirements

For detailed requirements and rubric criteria, see `project-task-description.md`.

### Key Requirements Summary

- **Python version**: 3.13 (must match in development and CI/CD)
- **Column names**: Contain hyphens - use Pydantic Field aliases to handle them
- **Model files**: (*.pkl, *.joblib) should be committed to GitHub for production use
- **Testing**: Minimum 6 tests by project completion (3+ for model, 3 for API)
- **CI/CD**: Must run pytest and flake8 on every push, both must pass
- **Screenshots required**:
  - `continuous_integration.png` - Successful CI run
  - `example.png` - API documentation showing example
  - `continuous_deployment.png` - CD enabled on Render
  - `live_get.png` - Browser showing GET endpoint
  - `live_post.png` - POST request result
- **Output files required**:
  - `slice_output.txt` - Model performance on data slices

## Deployment on Render

Render is a cloud platform that offers a free tier for small projects, making it ideal for learning and development. It provides automatic deployments from GitHub and supports FastAPI applications out of the box.

### Setting up Render

1. **Create a Render account**: Sign up at [render.com](https://render.com) (free tier available)

2. **Create a new Web Service**:
   - Connect your GitHub repository
   - Select the repository and branch
   - Choose "Web Service" as the service type

3. **Configure the service**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Python 3.13
   - **Plan**: Free (for development/testing)

4. **Environment Variables** (if needed):
   - Add any required environment variables in the Render dashboard
   - For AWS/DVC access, configure `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`

5. **Automatic Deployments**:
   - Render automatically deploys on every push to the connected branch
   - You can enable "Auto-Deploy" in the service settings

### Render vs. Heroku

- **Free Tier**: Render offers a free tier that doesn't require a credit card
- **Automatic HTTPS**: SSL certificates are automatically provisioned
- **GitHub Integration**: Seamless integration with GitHub for CI/CD
- **Sleep Mode**: Free tier services sleep after 15 minutes of inactivity (wakes on first request)

### Important Notes for Render Deployment

- Use `0.0.0.0` as the host in uvicorn to bind to all interfaces
- Use `$PORT` environment variable (provided by Render) for the port
- Ensure all dependencies are listed in `requirements.txt`
- Model files should be committed to the repository or loaded from external storage (S3, etc.)

For detailed German documentation, see the `/docs` folder.

## Documentation

Detailed German documentation for beginners is available in the `/docs` folder. This documentation includes:
- Background knowledge and concepts
- Step-by-step explanations
- Detailed implementation guides
- Troubleshooting tips

## Dataset Information

The Census dataset is used for income prediction. More information can be found [here](https://archive.ics.uci.edu/ml/datasets/census+income).
