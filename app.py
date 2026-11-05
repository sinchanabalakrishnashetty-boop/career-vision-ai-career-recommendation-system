import os
import logging
import io
from flask import Flask, render_template, request, jsonify, session, url_for, redirect, flash
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename
import torch
import pdfplumber



from career_recommender import CareerRecommender
from dqn_agent import DQNAgent
import utils

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "career-vision-ai-secret")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Initialize CSRF protection
csrf = CSRFProtect(app)

# Configure database with fallback
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", "sqlite:///career_vision.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize SQLAlchemy **after** app config
db = SQLAlchemy(app)

# Set up logging
logger = logging.getLogger(__name__)

# Initialize career recommender and DQN agent placeholders
career_recommender = None
dqn_agent = None

with app.app_context():
    # Import models to ensure tables are created
    import models
    db.create_all()

    # Initialize career recommender and DQN agent
    career_list = utils.get_career_list()
    career_recommender = CareerRecommender()
    embedding_dim = career_recommender.embedding_dim

    dqn_agent = DQNAgent(
        state_dim=embedding_dim,
        action_dim=len(career_list),
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        memory_size=10000,
        batch_size=64
    )

    # Try to load the model if it exists
    try:
        dqn_agent.load_model("dqn_model.pth")
        logger.info("Loaded existing DQN model")
    except FileNotFoundError:
        logger.info("No existing model found, using new DQN model")

# Utility functions for file handling
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf', 'docx'}

def extract_text_from_pdf(file_path):
    import pdfplumber
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        text = f"Error extracting text from PDF: {str(e)}"
    return text

def extract_text_from_docx(file_path):
    from docx import Document
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {str(e)}")
        text = f"Error extracting text from DOCX: {str(e)}"
    return text

# ----------------------- ROUTES -----------------------

@app.route('/')
def index():
    return render_template('index.html', resume_text="")

@app.route('/analyze', methods=['POST'])
def analyze():
    resume_text = request.form.get('resume_text', '')
    if not resume_text:
        flash("Please provide a resume or skills description.", "danger")
        return redirect(url_for('index'))
    try:
        resume_embedding = career_recommender.get_embedding(resume_text)
        action = dqn_agent.select_action(resume_embedding)
        career_list = utils.get_career_list()
        recommended_career = career_list[action]

        session['resume_embedding'] = resume_embedding.tolist()
        session['recommended_action'] = int(action)

        career_info = utils.get_career_info(recommended_career)

        return render_template('results.html',
                               career=recommended_career,
                               career_info=career_info)
    except Exception as e:
        logger.error(f"Error analyzing resume: {str(e)}")
        flash(f"An error occurred while analyzing your resume: {str(e)}", "danger")
        return redirect(url_for('index'))

@app.route('/feedback', methods=['POST'])
def feedback():
    rating = int(request.form.get('rating', 3))
    if 'resume_embedding' not in session or 'recommended_action' not in session:
        flash("Session expired. Please submit your resume again.", "danger")
        return redirect(url_for('index'))
    try:
        reward_mapping = {1: -2.0, 2: -1.0, 3: 0.0, 4: 1.0, 5: 2.0}
        reward = reward_mapping[rating]

        state = torch.tensor(session['resume_embedding'], dtype=torch.float32)
        action = session['recommended_action']
        next_state = state

        dqn_agent.update(state, action, reward, next_state, False)
        loss = dqn_agent.train()
        dqn_agent.save_model("dqn_model.pth")

        new_feedback = models.UserFeedback(
            resume_text=request.form.get('resume_text', ''),
            recommended_career=utils.get_career_list()[action],
            rating=rating
        )
        db.session.add(new_feedback)

        buffer = io.BytesIO()
        torch.save(dqn_agent.q_network.state_dict(), buffer)
        buffer.seek(0)
        new_model_state = models.ModelState(
            model_data=buffer.read(),
            epsilon=dqn_agent.epsilon,
            loss=loss
        )
        db.session.add(new_model_state)
        db.session.commit()

        flash("Thank you for your feedback! Our AI has learned from your response.", "success")
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        flash(f"Error: {str(e)}", "danger")
        return redirect(url_for('index'))

@app.route('/stats')
def stats():
    feedback_stats = db.session.query(
        models.UserFeedback.recommended_career,
        db.func.avg(models.UserFeedback.rating).label('avg_rating'),
        db.func.count(models.UserFeedback.id).label('count')
    ).group_by(models.UserFeedback.recommended_career).all()

    total_feedback = db.session.query(models.UserFeedback).count()
    avg_rating = db.session.query(db.func.avg(models.UserFeedback.rating)).scalar() or 0

    stats_data = {
        'total_feedback': total_feedback,
        'avg_rating': round(float(avg_rating), 2),
        'career_stats': [{
            'career': stat.recommended_career,
            'avg_rating': round(float(stat.avg_rating), 2),
            'count': stat.count
        } for stat in feedback_stats]
    }
    return render_template('stats.html', stats=stats_data)

# ----------------------- ADMIN ROUTES -----------------------

@app.route('/admin')
def admin_dashboard():
    model_states = db.session.query(models.ModelState).order_by(models.ModelState.timestamp.desc()).limit(20).all()
    epsilon_data = {
        'labels': [state.timestamp.strftime('%Y-%m-%d %H:%M') for state in model_states][::-1],
        'values': [float(state.epsilon) for state in model_states][::-1]
    } if model_states else {'labels': [], 'values': []}

    loss_states = [s for s in model_states if s.loss is not None]
    loss_data = {
        'labels': [s.timestamp.strftime('%Y-%m-%d %H:%M') for s in loss_states][::-1],
        'values': [float(s.loss) for s in loss_states][::-1]
    } if loss_states else {'labels': [], 'values': []}

    career_distribution = db.session.query(
        models.UserFeedback.recommended_career,
        db.func.count(models.UserFeedback.id).label('count')
    ).group_by(models.UserFeedback.recommended_career).all()

    career_dist_data = {
        'labels': [item.recommended_career for item in career_distribution],
        'values': [int(item.count) for item in career_distribution]
    } if career_distribution else {'labels': [], 'values': []}

    total_feedback = db.session.query(models.UserFeedback).count()
    avg_rating = db.session.query(db.func.avg(models.UserFeedback.rating)).scalar() or 0
    current_epsilon = dqn_agent.epsilon if dqn_agent else 1.0

    stats_data = {
        'total_feedback': total_feedback,
        'avg_rating': round(float(avg_rating), 2),
        'current_epsilon': current_epsilon
    }

    return render_template('admin.html',
                           stats=stats_data,
                           epsilon_data=epsilon_data,
                           loss_data=loss_data,
                           career_dist_data=career_dist_data,
                           model_states=model_states)

# ----------------------- UPLOAD ROUTES -----------------------

@app.route('/upload-resume', methods=['POST'])
def upload_resume():
    if 'resume_file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('index'))

    file = request.files['resume_file']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        filename = secure_filename(file.filename)
        filepath = os.path.join(temp_dir, filename)
        file.save(filepath)

        try:
            text = extract_text_from_pdf(filepath) if filename.lower().endswith('.pdf') else extract_text_from_docx(filepath)
            os.remove(filepath)

            if not text.strip():
                flash('Could not extract text from the uploaded file.', 'warning')
                return redirect(url_for('index'))

            return render_template('index.html', resume_text=text)
        except Exception as e:
            logger.error(f"Error processing uploaded file: {str(e)}")
            flash(f"Error processing the file: {str(e)}", 'danger')
            if os.path.exists(filepath):
                os.remove(filepath)
            return redirect(url_for('index'))

    flash('Invalid file type. Please upload a PDF or DOCX file.', 'danger')
    return redirect(url_for('index'))

@app.route('/update-agent-config', methods=['POST'])
def update_agent_config():
    try:
        learning_rate = float(request.form.get('learning_rate', 0.001))
        epsilon_decay = float(request.form.get('epsilon_decay', 0.995))
        batch_size = int(request.form.get('batch_size', 64))

        if not (0.0001 <= learning_rate <= 0.1):
            flash("Learning rate must be between 0.0001 and 0.1", "danger")
            return redirect(url_for('admin_dashboard'))
        if not (0.9 <= epsilon_decay <= 0.999):
            flash("Epsilon decay must be between 0.9 and 0.999", "danger")
            return redirect(url_for('admin_dashboard'))
        if not (8 <= batch_size <= 256):
            flash("Batch size must be between 8 and 256", "danger")
            return redirect(url_for('admin_dashboard'))

        if dqn_agent:
            for param_group in dqn_agent.optimizer.param_groups:
                param_group['lr'] = learning_rate
            dqn_agent.epsilon_decay = epsilon_decay
            dqn_agent.batch_size = batch_size
            dqn_agent.save_model("dqn_model.pth")
            flash("Agent configuration updated successfully", "success")
        else:
            flash("DQN agent not initialized", "danger")

        return redirect(url_for('admin_dashboard'))
    except Exception as e:
        logger.error(f"Error updating agent config: {str(e)}")
        flash(f"Error updating agent config: {str(e)}", "danger")
        return redirect(url_for('admin_dashboard'))

# ----------------------- RUN APP -----------------------
if __name__ == "__main__":
    app.run(debug=True)
