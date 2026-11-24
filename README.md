# Smart Hospital – Patient Waiting Time Predictor

A BCA minor-project prototype that estimates patient queue waiting time using a Random Forest Regression model and displays analytics in Streamlit.

## Features
- Patient queue input form
- Machine-learning waiting-time estimate
- Queue status indicator
- Department analytics and charts
- Local CSV prediction history
- CSV download
- Project explanation and model workflow

## Important project note
The model is trained on **synthetic demonstration data**, not real hospital records. Results are illustrative and must not be used for real patient scheduling, clinical decisions, or emergency triage. A production deployment requires approved representative data, validation, privacy/security controls, and operational review.

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Start the app:

   ```bash
   streamlit run app.py
   ```

6. Open the local URL printed in the terminal (usually http://localhost:8501).

## Deploy using Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and `README.md`.
2. Sign in to Streamlit Community Cloud using GitHub.
3. Choose **Create app**, select the repository and branch, and set the main file path to `app.py`.
4. Click **Deploy**.

The deployment provider may require account verification or access approval. The local CSV history is not a durable database and may reset on hosted deployments; use a persistent database for a real production service.

## ML workflow
Input features:
- Department
- Patients ahead in queue
- Available doctors
- Average consultation duration
- Hour of day
- Weekday/weekend

Target:
- Estimated waiting time in minutes

Algorithm:
- Random Forest Regressor with one-hot encoding for department.

## Suggested viva explanation
"We use supervised machine learning because the model learns from example input-output pairs. Waiting time is a continuous numerical value, so this is a regression problem. Random Forest combines predictions from multiple decision trees to improve robustness. The dashboard also performs basic data mining by grouping prediction history by department and showing patterns."

## Suggested future enhancements
- Train and validate with anonymized, approved real hospital queue data.
- Add login and role-based access.
- Store records in a persistent database.
- Track model error (MAE/RMSE) on a separate test set.
- Add queue updates and staff verification.
