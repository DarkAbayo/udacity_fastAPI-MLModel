# FastAPI ML Project Index

## Project Overview
This is a machine learning project for deploying a Census income prediction model using FastAPI on a cloud platform (Render). The project is part of a training exercise.

## Project Structure

### Source Locations
- **Starter Code**: `/nd0821-c3-starter-code/starter/`
- **Project Directory**: `/project/` (current working directory)

### Key Components

#### 1. Data Processing (`starter/ml/data.py`)
- Function: `process_data()` - Handles one-hot encoding for categorical features and label binarization
- Input: DataFrame with categorical features and optional label
- Output: Processed numpy arrays, encoder, and label binarizer

#### 2. Model Training (`starter/ml/model.py`)
- Functions:
  - `train_model()` - Trains ML model (to be implemented)
  - `compute_model_metrics()` - Calculates precision, recall, F1 score
  - `inference()` - Runs model predictions (to be implemented)
- Model type: RandomForestClassifier (based on docstrings)

#### 3. Training Script (`starter/train_model.py`)
- Entry point for model training
- Uses train-test split
- Categorical features defined: workclass, education, marital-status, occupation, relationship, race, sex, native-country
- Label: "salary"

#### 4. API (`starter/main.py`)
- Currently empty - needs FastAPI implementation
- Requirements:
  - GET endpoint on root (welcome message)
  - POST endpoint for model inference
  - Type hinting required
  - Pydantic model for POST body (with example)
  - Handle column names with hyphens

#### 5. Data
- Location: `data/census.csv`
- Note: Data is messy and needs cleaning (remove spaces)

#### 6. Documentation
- Location: `docs/` folder
- Language: German
- Purpose: Detailed, beginner-friendly documentation with background knowledge
- Format: Markdown (.md) files
- Content: Step-by-step guides, concepts, implementation details, troubleshooting

#### 7. Dependencies
- FastAPI, uvicorn, pydantic
- scikit-learn, pandas, numpy
- pytest, httpx, requests
- Python 3.13

## Project Requirements

For complete requirements and rubric, see `project-task-description.md`.

### Model Development
- [ ] Complete `train_model()` function (train, save, load model and encoders)
- [ ] Complete `inference()` function
- [ ] Write training script that uses implemented functions
- [ ] Write unit tests for at least 3 model functions (test return types)
- [ ] Write function for performance on data slices
  - Output to `slice_output.txt`
  - Compute metrics for each unique categorical value
- [ ] Create model card using template
  - Address every section
  - Include metrics and performance
  - Complete sentences

### API Development
- [ ] Implement GET endpoint (welcome message on root)
- [ ] Implement POST endpoint (model inference on different path)
- [ ] Create Pydantic model for POST body with example
- [ ] Use type hints for automatic FastAPI documentation
- [ ] Handle column names with hyphens (Pydantic Field aliases)
- [ ] Write 3 unit tests:
  - 1 GET (test status code AND content)
  - 2 POST (one for each possible prediction)
- [ ] Run `python sanitycheck.py` and fix issues
- [ ] Screenshot: `example.png` (API docs with example)

### Git & CI/CD
- [ ] Set up GitHub repository
- [ ] Configure GitHub Actions:
  - Run pytest and flake8 on every push to main/master
  - Use Python 3.13
  - Both must pass without errors
  - Minimum 6 tests by completion
- [ ] Screenshot: `continuous_integration.png`

### Deployment
- [ ] Deploy to Render (auto-deploy from GitHub)
- [ ] Configure environment variables on Render (if needed)
- [ ] Screenshot: `continuous_deployment.png`
- [ ] Screenshot: `live_get.png` (browser showing GET)
- [ ] Write script to POST to live API using requests
- [ ] Screenshot: `live_post.png` (POST result)

### Documentation
- [ ] Create German documentation in `/docs` folder
- [ ] Document background concepts and theory
- [ ] Provide step-by-step implementation guides
- [ ] Include troubleshooting and FAQ sections

## File Structure

```
project/
├── starter/
│   ├── __init__.py
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── data.py
│   │   └── model.py
│   └── train_model.py
├── data/
│   └── census.csv
├── docs/                   # German documentation (beginner-friendly)
├── main.py
├── requirements.txt
├── setup.py
├── sanitycheck.py
├── model_card_template.md
├── project-task-description.md  # Detailed requirements and rubric
├── .gitignore
└── README.md
```

## Deployment Platform: Render

- **Platform**: Render (free tier available, no credit card required)
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Build Command**: `pip install -r requirements.txt`
- **Environment**: Python 3.13
- **Auto-deploy**: Enabled for GitHub pushes
- **Sleep Mode**: Free tier services sleep after 15 min inactivity

## Documentation

- **Location**: `/docs` folder
- **Language**: German (exception to standard cursorrules for learning purposes)
- **Target Audience**: Beginners learning ML deployment
- **Content**: Detailed explanations with background knowledge, step-by-step guides, troubleshooting
- **Format**: Markdown (.md) files
- **Note**: While cursorrules specify English for documentation, the `/docs` folder uses German to provide detailed beginner-friendly explanations with necessary background knowledge for learning purposes.

## Deliverables Checklist

### Required Files
- `slice_output.txt` - Model performance on data slices
- `continuous_integration.png` - Successful CI run
- `example.png` - API documentation with example
- `continuous_deployment.png` - CD enabled on Render
- `live_get.png` - Browser showing GET endpoint
- `live_post.png` - POST request result

### Required Code
- Training script using implemented functions
- Model card (complete, using template)
- API with GET and POST endpoints
- Minimum 6 tests total (3+ model, 3 API)

## Notes
- Python version: 3.13 (must match in dev and CI/CD)
- Column names contain hyphens - use Pydantic Field aliases
- Model files (*.pkl, *.joblib) should be committed to GitHub for production
- Use DVC for data versioning (optional, depending on project needs)
- CI/CD should run pytest and flake8 on Python 3.13
- Environment setup: pip and venv (recommended approach)
- Run `python sanitycheck.py` before final submission

