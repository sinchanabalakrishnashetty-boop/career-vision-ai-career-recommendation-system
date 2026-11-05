# CareerVisionAI

<p align="center">
  <img src="generated-icon.png" alt="CareerVisionAI logo" width="160" />
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img alt="Flask" src="https://img.shields.io/badge/Flask-3.1-000000?style=for-the-badge&logo=flask&logoColor=white" />
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.7-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img alt="SQLite" src="https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white" />
</p>

CareerVisionAI is an intelligent career recommendation web application that analyzes resume content and skill descriptions to suggest suitable career paths using keyword-based feature extraction and a Deep Q-Network (DQN).

## Overview

CareerVisionAI helps users explore career opportunities based on their education, skills, experience, and resume content. Users can either paste résumé text manually or upload a PDF/DOCX file. The app extracts relevant information, converts it into a numerical feature vector, and recommends one of 25 predefined career paths.

The system also supports user feedback. Each recommendation can be rated from 1 to 5, and the rating is converted into a reward signal that helps improve the recommendation model over time.

## Key Features

- Resume text input and file upload support for PDF and DOCX
- Keyword-based skill and education analysis
- 329-dimensional feature extraction from resume text
- PyTorch-based Deep Q-Network recommendation engine
- 25 predefined career options
- Feedback-based learning using user ratings
- Recommendation statistics and learning dashboard
- Admin dashboard for monitoring epsilon, loss, and feedback distribution

## Technology Stack

- Python 3.11+
- Flask
- Jinja2 Templates
- SQLAlchemy
- SQLite (default) / PostgreSQL-ready
- PyTorch
- NumPy
- pdfplumber
- python-docx
- Flask-WTF
- Chart.js
- Gunicorn

## System Architecture

The project follows a Flask-based web architecture:

### 1. Presentation Layer
- HTML
- CSS
- Bootstrap
- JavaScript
- Jinja templates

### 2. Application Layer
- Flask app routes
- Resume processing
- Career recommendation logic
- Feedback handling

### 3. Feature Extraction Layer
- Text normalization
- Keyword-based skill detection
- Education/experience analysis
- Numerical feature vector generation

### 4. Recommendation Layer
- DQN model
- Q-value estimation
- Epsilon-greedy action selection
- Reward-based learning

### 5. Data Layer
- SQLAlchemy models
- SQLite default database
- Optional PostgreSQL configuration via `DATABASE_URL`

## Recommendation Workflow

1. User enters resume text or uploads a PDF/DOCX file.
2. The app extracts readable text from the document.
3. Resume text is normalized and transformed into a 329-feature vector.
4. The DQN evaluates possible careers using Q-values.
5. The best-fit career is displayed to the user.
6. The user rates the recommendation.
7. The rating is converted into a reward and used to retrain the model when enough experience exists.
8. Feedback and model information are saved in the database.

## Recommendation Model Details

### Feature Vector
The recommender generates 329 numerical features across three categories:

- 306 skill keyword features
- 10 general text features
- 13 domain-level features

### DQN Architecture

| Component | Configuration |
|---|---|
| Input size | 329 |
| Hidden layer 1 | 256 neurons |
| Hidden layer 2 | 128 neurons |
| Hidden layer 3 | 64 neurons |
| Output layer | 25 career actions |
| Activation | ReLU |
| Optimizer | Adam |
| Loss function | MSE |
| Learning rate | 0.001 |
| Discount factor | 0.99 |
| Replay buffer | 10,000 |
| Batch size | 64 |

### Feedback Reward Mapping

| User Rating | Reward |
|---|---|
| 1 | -2 |
| 2 | -1 |
| 3 | 0 |
| 4 | +1 |
| 5 | +2 |

## Career Options

The app recommends from these 25 career paths:

- Data Scientist
- Software Engineer
- AI Engineer
- Machine Learning Engineer
- Product Manager
- UX Designer
- DevOps Engineer
- Cloud Architect
- Cybersecurity Analyst
- Business Analyst
- Web Developer
- Mobile App Developer
- Database Administrator
- Network Engineer
- Project Manager
- Digital Marketing Specialist
- Systems Administrator
- Data Engineer
- Full Stack Developer
- QA Engineer
- Technical Writer
- Blockchain Developer
- AR/VR Developer
- Game Developer
- IoT Engineer

## Project Structure

```text
CareerVisionAI/
├── app.py
├── main.py
├── models.py
├── career_recommender.py
├── dqn_agent.py
├── utils.py
├── dqn_model.pth
├── pyproject.toml
├── uv.lock
├── README.md
├── SETUP.md
├── generated-icon.png
├── attached_assets/
├── instance/
├── static/
│   ├── css/
│   └── js/
├── templates/
│   ├── admin.html
│   ├── index.html
│   ├── layout.html
│   ├── results.html
│   └── stats.html
├── temp_uploads/
└── __pycache__/
```

## Routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Home page and resume input form |
| `/analyze` | POST | Analyze the resume and recommend a career |
| `/upload-resume` | POST | Extract text from uploaded PDF/DOCX files |
| `/feedback` | POST | Save rating and train the model |
| `/stats` | GET | Display recommendation metrics |
| `/admin` | GET | View admin dashboard and training insights |
| `/update-agent-config` | POST | Update learning parameters |

## Database Design

The app uses SQLAlchemy models to store:

- `UserFeedback`: resume text, recommended career, rating, and timestamp
- `ModelState`: model state, epsilon, timestamp, and loss history

Default database: SQLite

Optional database: PostgreSQL via `DATABASE_URL`

## Getting Started

### Prerequisites

- Python 3.11 or newer
- Git
- pip or uv

### Clone the Repository

```bash
git clone <your-repository-url>
cd CareerVisionAI
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Environment

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### Install Dependencies

Using uv:

```bash
pip install uv
uv sync
```

Or using pip:

```bash
pip install flask flask-login flask-sqlalchemy flask-wtf gunicorn psycopg2-binary torch numpy pdfplumber python-docx email-validator werkzeug
```

### Run the Application

```bash
python main.py
```

Then open:

```text
http://127.0.0.1:5000/
```

## Demo / Screenshots

> Screenshots can be added here as the project grows.

## Learning Outcomes

This project helped explore:

- Python web application development
- Flask routing and form handling
- Resume document processing
- PDF and DOCX text extraction
- Feature engineering for text-based recommendations
- Deep Q-Learning fundamentals
- PyTorch neural network training
- Experience replay and reward-based learning
- SQLAlchemy database modeling
- Dashboard development and visualization

## Future Enhancements

- Improve resume parsing with modern NLP techniques
- Add stronger skill matching and semantic understanding
- Expand career information with up-to-date labor market data
- Add user authentication for the admin dashboard
- Improve resume privacy and data security
- Add recommendation explanations for each result
- Add model evaluation metrics and richer analytics

## Contributing

Contributions are welcome. If you want to improve the system:

1. Fork the project.
2. Create a feature branch.
3. Commit your improvements.
4. Open a pull request with a clear summary.

## Acknowledgements

This project demonstrates how reinforcement learning can be applied to practical recommendation tasks beyond games and simulations.

---

Thank you for visiting CareerVisionAI.
