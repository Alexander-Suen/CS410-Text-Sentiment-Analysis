# Text-Based Sentiment Analysis for Product Reviews


### Project Description
Github Repo: https://github.com/Alexander-Suen/CS410-Text-Sentiment-Analysis/

Report: https://docs.google.com/document/d/10rc6Pk_CcjJO9RlF_7TWrIVctYKhaMIpQHJr4AGsSOQ/edit?usp=sharing

Presentation: https://docs.google.com/presentation/d/1XxK8e_-8g96gtWm81ytv_tIMtaDbK3cdS_Z_y1UADWI/edit?usp=sharing

## Project Overview

**Course**: CS410 Text Information Systems - Fall 2024 

**Project Title**: Text-Based Sentiment Analysis for Product Reviews

**Team Members**:

- Prithvi Balaji (Project Coordinator, GitHub: `pbalaji3`)
- Avinash Tiwari (GitHub: `tiwari6`)
- Brendan Heaney (GitHub: `bheaney2`)
- Minghao Sun (GitHub: `msun60`)


### Project Description

This project aims to develop a sentiment analysis tool specifically for product reviews using Natural Language Processing (NLP) techniques. The primary objectives are:

- To classify reviews as positive, negative, or neutral.
- To analyze customer feedback, extracting key themes and sentiments to gain insights.
- To create a web application that allows users to input reviews and obtain sentiment analysis results in real-time.

The tool is developed using Python, leveraging libraries such as NLTK, TextBlob, or spaCy for NLP tasks. Machine learning libraries such as Scikit-learn, TensorFlow, or PyTorch are used for building and training models. The final application will be deployed as a web interface using Django or Flask frameworks.

------

## Progress Reports

### Week 1–2: Project Setup and Data Collection

- **Completed Tasks**:
  - Created GitHub repository and set up project structure.
  - Assigned initial roles and responsibilities within the team.
  - Collected product review data from Amazon using BeautifulSoup for web scraping.
  - Cleaned data by removing stop words, punctuation, and tokenizing text using NLTK.
- **Pending Tasks**:
  - Further preprocessing, including stemming and lemmatization.
  - Begin feature engineering and model selection.
- **Challenges**:
  - Encountered issues with data scraping limitations; considering using open-source datasets if needed.
  - Some data inconsistencies in initial cleaning results; refining cleaning process to improve accuracy.

------

### Week 3–4: Feature Engineering and Model Selection

- **Completed Tasks**:
  - Converted text data to numerical format using TF-IDF.
  - Explored word embeddings (Word2Vec and GloVe) to capture contextual meanings.
  - Experimented with baseline models (Naive Bayes, SVM).
- **Pending Tasks**:
  - Finalize model selection and tune hyperparameters.
  - Begin implementing deep learning models (LSTM) if initial models are insufficient.
- **Challenges**:
  - Difficulty achieving high accuracy with initial models; considering additional preprocessing steps to improve results.

------

### Week 5–6: Model Optimization and Evaluation

- **Completed Tasks**:
  - Fine-tuned selected models with grid search on hyperparameters.
  - Evaluated models using accuracy, precision, recall, and F1-score.
  - Began incorporating BERT-based model to handle contextual nuances.
- **Pending Tasks**:
  - Complete evaluation and finalize the best-performing model for deployment.
- **Challenges**:
  - BERT model requires significant computational resources; adjusting model settings to improve efficiency.

------

### Week 7–8: Web Application Development

- **Completed Tasks**:
  - Backend setup using Flask for API integration with sentiment model.
  - Basic frontend interface design completed, allowing users to input reviews.
- **Pending Tasks**:
  - Integrate model predictions with the frontend.
  - Complete frontend styling and user experience adjustments.
- **Challenges**:
  - Synchronizing backend and frontend data flow; testing required to ensure smooth functionality.

------

### Week 9: Testing and Final Adjustments

- **Completed Tasks**:
  - Initial testing of the web application with sample data.
  - Gathered feedback on user experience and model accuracy.
- **Pending Tasks**:
  - Fix identified bugs and improve UI based on feedback.
  - Prepare for final presentation and documentation.

------

### Week 10: Final Presentation and Documentation

- **Completed Tasks**:
  - Documented each phase of development, including data preprocessing, model training, and deployment steps.
  - Prepared slides and final report for project submission.
- **Pending Tasks**:
  - Final adjustments based on any last-minute feedback.
  - Record video presentation (if applicable) and upload to GitHub.

------

## Usage Instructions

### Prerequisites

- Python 3.x
- Libraries listed in `requirements.txt`

### Setup

1. Clone the repository:

   ```
   git clone https://github.com/your-username/Text-Sentiment-Analysis.git
   cd Text-Sentiment-Analysis
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

### Running the Application

1. Run the backend server:

   ```
   python manage.py runserver
   ```

2. Access the web interface at `http://localhost:8000`.

------

## Additional Resources

- **Documentation**: Comprehensive code documentation is available in the `docs/` directory.
- **Presentation**: A link to the final presentation or video is provided in the repository.
