import os
from datetime import datetime, date
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ------------------------------------------------------------
# SMART HOSPITAL: six-feature educational prototype
# ------------------------------------------------------------
st.set_page_config(
    page_title="Smart Hospital Predictor",
    page_icon="🏥",
    layout="wide"
)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

PATIENTS_FILE = os.path.join(DATA_DIR, "patients.csv")
DOCTORS_FILE = os.path.join(DATA_DIR, "doctors.csv")
HISTORY_FILE = os.path.join(DATA_DIR, "prediction_history.csv")

DEPARTMENTS = [
    "General Medicine", "Cardiology", "Orthopedics",
    "Pediatrics", "Dermatology", "ENT"
]
PRIORITIES = ["Normal", "Moderate", "High", "Emergency"]

PATIENT_COLUMNS = [
    "token", "patient_name", "age", "phone", "department",
    "symptoms", "priority", "status", "registered_at", "assigned_doctor_id",
    "hospital_name"
]
DOCTOR_COLUMNS = [
    "doctor_id", "doctor_name", "department", "availability",
    "consultation_minutes", "hospital_name"
]
HISTORY_COLUMNS = [
    "token", "department", "patients_ahead", "available_doctors",
    "avg_consultation_min", "hour", "is_weekend", "priority",
    "predicted_wait_min", "queue_status", "created_at"
]


def ensure_csv(path, columns, default_df=None):
    """Create a CSV with headers if it does not exist."""
    if not os.path.exists(path):
        df = default_df.copy() if default_df is not None else pd.DataFrame(columns=columns)
        df.to_csv(path, index=False)


def read_csv_safe(path, columns):
    ensure_csv(path, columns)
    try:
        df = pd.read_csv(path)
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
        # Doctor IDs are text values (for example D001), never numeric.
        # Old CSV files may contain an all-empty column inferred as float64.
        if "assigned_doctor_id" in df.columns:
            df["assigned_doctor_id"] = (
                df["assigned_doctor_id"].astype("string").fillna("").astype(str)
            )
        if "hospital_name" in df.columns:
            df["hospital_name"] = (
                df["hospital_name"].astype("string").fillna("").astype(str)
            )
        return df[columns]
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=columns)


def save_df(df, path):
    df.to_csv(path, index=False)


# Add sample doctors only when the doctor file is first created.
# Publicly listed Ranchi doctors. Availability values below are demonstration
# defaults only; they are NOT live hospital schedules or real-time availability.
sample_doctors = pd.DataFrame([
    ["P001", "Dr. Awkash Kumar", "Orthopedics", "Available", 20, "Paras HEC Hospital"],
    ["P002", "Dr. Ashutosh Kumar Thakur", "General Medicine", "Available", 15, "Paras HEC Hospital"],
    ["P003", "Dr. Gunjesh Kumar Singh", "Oncology", "Available", 25, "Paras HEC Hospital"],
    ["P004", "Dr. Kunal Kumar Singh", "Urology", "Available", 20, "Paras HEC Hospital"],
    ["O001", "Dr. Muneshwar Kumar", "Cardiology", "Available", 20, "Orchid Medical Centre"],
    ["O002", "Dr. Kunwer Abhishek Ary", "Cardiology", "Available", 20, "Orchid Medical Centre"],
    ["O003", "Dr. Shanti Mala Tirkey", "Cardiology", "Available", 20, "Orchid Medical Centre"],
    ["S001", "Dr. Ashutosh Charan", "General Medicine", "Available", 15, "Santevita Hospital"],
    ["S002", "Dr. Mohit Narayan", "General Medicine", "Available", 15, "Santevita Hospital"],
    ["S003", "Dr. R. K. Maroo", "General Surgery", "Available", 20, "Santevita Hospital"],
], columns=DOCTOR_COLUMNS)

ensure_csv(PATIENTS_FILE, PATIENT_COLUMNS)
ensure_csv(DOCTORS_FILE, DOCTOR_COLUMNS, sample_doctors)
ensure_csv(HISTORY_FILE, HISTORY_COLUMNS)

# ------------------------------------------------------------
# Original Random Forest model retained
# ------------------------------------------------------------
@st.cache_resource
def train_model():
    """Train a demonstration model on synthetic hospital queue data."""
    rng = np.random.default_rng(42)
    n = 2200
    dept = rng.choice(DEPARTMENTS, n)
    patients = rng.integers(0, 31, n)
    doctors = rng.integers(1, 6, n)
    avg_consult = rng.integers(8, 26, n)
    hour = rng.integers(8, 21, n)
    weekend = rng.integers(0, 2, n)

    dept_effects = {
        "General Medicine": 2, "Cardiology": 8, "Orthopedics": 5,
        "Pediatrics": 3, "Dermatology": -2, "ENT": 1
    }
    department_effect = np.array([dept_effects[d] for d in dept])
    peak_load = (
        np.where((hour >= 10) & (hour <= 13), 7, 0)
        + np.where((hour >= 17) & (hour <= 19), 4, 0)
    )
    weekend_load = weekend * 3
    wait = (
        4
        + (patients * avg_consult / doctors) * 0.72
        + department_effect
        + peak_load
        + weekend_load
        + rng.normal(0, 5, n)
    )
    wait = np.clip(wait, 2, 240)

    X = pd.DataFrame({
        "department": dept,
        "patients_ahead": patients,
        "available_doctors": doctors,
        "avg_consultation_min": avg_consult,
        "hour": hour,
        "is_weekend": weekend
    })
    y = wait

    preprocessor = ColumnTransformer([
        ("department", OneHotEncoder(handle_unknown="ignore"), ["department"]),
        ("numbers", "passthrough", [
            "patients_ahead", "available_doctors",
            "avg_consultation_min", "hour", "is_weekend"
        ])
    ])
    model = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=140, random_state=42, min_samples_leaf=2
        ))
    ])
    model.fit(X, y)
    return model, X, y


model, demo_X, demo_y = train_model()


def priority_note(priority):
    notes = {
        "Normal": "Routine queue category.",
        "Moderate": "Review according to hospital policy.",
        "High": "Please ask authorized clinical staff to review promptly.",
        "Emergency": "Alert hospital staff immediately and follow the hospital's emergency protocol."
    }
    return notes.get(priority, "")


def get_queue_counts(patients, department):
    active = patients[
        (patients["department"] == department)
        & (patients["status"].isin(["Waiting", "In Consultation"]))
    ]
    return len(active), active


def next_token(patients):
    if patients.empty or patients["token"].dropna().empty:
        return "P001"
    nums = []
    for value in patients["token"].astype(str):
        digits = "".join(ch for ch in value if ch.isdigit())
        if digits:
            nums.append(int(digits))
    return f"P{(max(nums, default=0) + 1):03d}"


def assign_doctor_at_registration(patients, doctors_df, patient_index):
    """Reserve an available doctor in the same department for a new waiting patient.

    Reservation is stored on the patient record. The doctor's status remains
    Available until consultation actually starts, when the queue workflow marks
    the assigned doctor Busy.
    """
    department = str(patients.loc[patient_index, "department"])
    hospital_name = str(patients.loc[patient_index].get("hospital_name", ""))
    already_assigned = patients.loc[
        (patients.index != patient_index)
        & (patients["department"].astype(str) == department)
        & (patients["hospital_name"].astype(str) == hospital_name)
        & (patients["status"].isin(["Waiting", "In Consultation"]))
        & (patients["assigned_doctor_id"].fillna("").astype(str).str.strip() != "")
    , "assigned_doctor_id"].astype(str).tolist()

    candidates = doctors_df[
        (doctors_df["department"].astype(str) == department)
        & (doctors_df["hospital_name"].astype(str) == hospital_name)
        & (doctors_df["availability"].astype(str) == "Available")
        & (~doctors_df["doctor_id"].astype(str).isin(already_assigned))
    ]

    if candidates.empty:
        patients.loc[patient_index, "assigned_doctor_id"] = ""
        return ""

    doctor_id = str(candidates.iloc[0]["doctor_id"])
    patients.loc[patient_index, "assigned_doctor_id"] = doctor_id
    return doctor_id


def assign_waiting_patients_for_department(patients, doctors_df, department):
    """Reserve newly available doctors for the oldest unassigned waiting patients."""
    if patients.empty or doctors_df.empty:
        return patients, doctors_df, []
    assignments = []
    waiting = patients[
        (patients["department"].astype(str) == str(department))
        & (patients["status"].astype(str) == "Waiting")
        & (patients["assigned_doctor_id"].fillna("").astype(str).str.strip() == "")
    ].sort_values("registered_at", kind="stable")

    for patient_index in waiting.index:
        doctor_id = assign_doctor_at_registration(patients, doctors_df, patient_index)
        if not doctor_id:
            break
        assignments.append((str(patients.loc[patient_index, "token"]), doctor_id))
    return patients, doctors_df, assignments


def update_doctor_for_queue_change(patients, doctors_df, patient_index, new_status):
    """Synchronize a doctor's status with a patient's consultation lifecycle.

    A doctor is assigned when a patient enters consultation and released when
    that patient leaves consultation. Only doctors assigned by this workflow
    are automatically released; manually Busy/Unavailable doctors are retained.
    """
    patient = patients.loc[patient_index]
    department = patient["department"]
    hospital_name = str(patient.get("hospital_name", ""))
    assigned_id = patient.get("assigned_doctor_id", np.nan)
    assigned_id = str(assigned_id) if pd.notna(assigned_id) and str(assigned_id).strip() else ""

    if new_status == "In Consultation":
        if assigned_id:
            matching = doctors_df[doctors_df["doctor_id"].astype(str) == assigned_id]
            if not matching.empty:
                doctors_df.loc[doctors_df["doctor_id"].astype(str) == assigned_id, "availability"] = "Busy"
                return doctors_df, assigned_id, None

        candidates = doctors_df[
            (doctors_df["department"] == department)
            & (doctors_df["hospital_name"].astype(str) == hospital_name)
            & (doctors_df["availability"] == "Available")
        ]
        if candidates.empty:
            return doctors_df, "", f"No available doctor in {department}. Patient status was not changed."
        doctor_id = str(candidates.iloc[0]["doctor_id"])
        doctors_df.loc[doctors_df["doctor_id"].astype(str) == doctor_id, "availability"] = "Busy"
        patients.loc[patient_index, "assigned_doctor_id"] = doctor_id
        return doctors_df, doctor_id, None

    # Leaving consultation: release the assigned doctor only when no other
    # patient is still assigned to that doctor and in consultation.
    if assigned_id:
        other_consultations = patients[
            (patients.index != patient_index)
            & (patients["assigned_doctor_id"].astype(str) == assigned_id)
            & (patients["status"] == "In Consultation")
        ]
        if other_consultations.empty:
            doctors_df.loc[
                (doctors_df["doctor_id"].astype(str) == assigned_id)
                & (doctors_df["availability"] == "Busy"),
                "availability"
            ] = "Available"
        patients.loc[patient_index, "assigned_doctor_id"] = ""

    return doctors_df, "", None


# ------------------------------------------------------------
# Premium visual theme and navigation (presentation layer only)
# ------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
:root { --navy:#10243A; --teal:#0F9D91; --mint:#E8F7F4; --ink:#172B42; --muted:#64748B; }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: linear-gradient(180deg,#F4F8FC 0%,#F8FAFC 48%,#F1F6FA 100%); color:var(--ink); }
[data-testid="stHeader"] { background: rgba(244,248,252,.88); }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#10243A 0%,#183B55 100%); }
[data-testid="stSidebar"] * { color:#F3F8FC !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { padding: .42rem .55rem; border-radius:10px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:rgba(255,255,255,.10); }
.block-container { padding-top:1.35rem; padding-bottom:3rem; max-width:1450px; }
.hero-shell { background:linear-gradient(115deg,#10243A 0%,#17445C 62%,#0F9D91 130%);
  border-radius:22px; padding:26px 30px; color:white; margin:0 0 22px 0;
  box-shadow:0 14px 34px rgba(16,36,58,.15); position:relative; overflow:hidden; }
.hero-shell:after { content:'✚'; position:absolute; right:38px; top:-34px; font-size:150px;
  color:rgba(255,255,255,.07); transform:rotate(12deg); }
.hero-kicker { color:#9DE5DC; font-size:12px; letter-spacing:2px; font-weight:800; text-transform:uppercase; }
.hero-title { font-size:32px; line-height:1.15; font-weight:800; margin:7px 0 5px; color:#fff; }
.hero-sub { color:#D7E7F0; font-size:14px; margin:0; }
.section-tag { color:#0F8F86; font-size:12px; font-weight:800; letter-spacing:1.2px; text-transform:uppercase; }
div[data-testid="stMetric"] { background:white; border:1px solid #E4EDF3; padding:15px 17px;
  border-radius:16px; box-shadow:0 5px 16px rgba(16,36,58,.045); }
div[data-testid="stMetricLabel"] { color:#64748B; font-weight:600; }
div[data-testid="stMetricValue"] { color:#10243A; font-weight:800; }
.stButton > button, .stDownloadButton > button, button[kind="primary"] {
  border-radius:11px; border:0; background:linear-gradient(100deg,#0F9D91,#147F89);
  color:white; font-weight:700; min-height:2.55rem; box-shadow:0 5px 12px rgba(15,157,145,.16); }
.stButton > button:hover, .stDownloadButton > button:hover { color:white; border:0; filter:brightness(.96); }
div[data-testid="stForm"], div[data-testid="stExpander"], div[data-testid="stDataFrame"],
div[data-testid="stPlotlyChart"] { border-radius:15px; }
div[data-testid="stForm"] { background:white; border:1px solid #E4EDF3; padding:18px; }
input, textarea, [data-baseweb="select"] > div { border-radius:10px !important; }
hr { border-color:#E2EAF1; }
div[data-testid="stAlert"] { border-radius:12px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-shell">
  <div class="hero-kicker">Smart care • smarter operations</div>
  <div class="hero-title">✚ Smart Hospital</div>
  <p class="hero-sub">AI-Based Patient Waiting Time Prediction & Hospital Operations Dashboard</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ✚ SMART HOSPITAL")
    st.caption("CARE OPERATIONS CONSOLE")
    page_icons = {
        "Patient Registration": "🧑‍⚕️  Patient Registration",
        "Queue Management": "🧾  Queue Management",
        "AI Waiting-Time Prediction": "⏱️  AI Waiting-Time Prediction",
        "Doctor Availability": "🩺  Doctor Availability",
        "Patient Priority Classification": "🚦  Patient Priority",
        "Admin Dashboard": "📊  Admin Dashboard",
        "About Project": "ℹ️  About Project"
    }
    page = st.radio(
        "NAVIGATION",
        list(page_icons.keys()),
        format_func=lambda value: page_icons[value]
    )
    st.divider()
    st.markdown("**⚙️ SYSTEM OVERVIEW**")
    st.caption("ML model · Random Forest Regression")
    st.caption("Interface · Streamlit")
    st.caption("Storage · Local CSV files")
    st.info(
        "Educational demonstration only. Clinical assessment and emergency decisions "
        "must remain with qualified hospital staff."
    )
    st.divider()
    st.write("**ML model:** Random Forest Regression")
    st.write("**Interface:** Streamlit")
    st.write("**Data storage:** Local CSV files")
    st.info(
        "Educational demonstration only. This system does not diagnose, "
        "triage, or replace decisions by qualified hospital staff."
    )

patients = read_csv_safe(PATIENTS_FILE, PATIENT_COLUMNS)
doctors_df = read_csv_safe(DOCTORS_FILE, DOCTOR_COLUMNS)
history = read_csv_safe(HISTORY_FILE, HISTORY_COLUMNS)
# Keep the curated Ranchi directory synchronized on every startup.
# Update only the known directory fields for seeded IDs; preserve live/demo
# availability values and every unrelated doctor row already in doctors.csv.
_seed_ids = set(sample_doctors["doctor_id"].astype(str))
for _, _seed in sample_doctors.iterrows():
    _mask = doctors_df["doctor_id"].astype(str) == str(_seed["doctor_id"])
    if _mask.any():
        for _col in ["doctor_name", "department", "hospital_name", "consultation_minutes"]:
            doctors_df.loc[_mask, _col] = _seed[_col]
    else:
        doctors_df = pd.concat([doctors_df, pd.DataFrame([_seed])], ignore_index=True)
doctors_df["doctor_id"] = doctors_df["doctor_id"].fillna("").astype(str).str.strip()
doctors_df["hospital_name"] = doctors_df["hospital_name"].fillna("").astype(str).str.strip()
doctors_df["department"] = doctors_df["department"].fillna("").astype(str).str.strip()
# Remove duplicate doctor IDs while preserving the first existing record.
doctors_df = doctors_df.drop_duplicates(subset=["doctor_id"], keep="first").reset_index(drop=True)
save_df(doctors_df, DOCTORS_FILE)

# ------------------------------------------------------------
# 1. Patient Registration
# ------------------------------------------------------------
if page == "Patient Registration":
    st.header("1. Patient Registration")
    st.write("Select a Ranchi hospital, department and preferred doctor. Public doctor names are included for demonstration; OPD timings and real-time availability must be confirmed with the hospital.")

    # IMPORTANT: Do not place cascading hospital/department/doctor selectors
    # inside st.form. Streamlit forms do not rerun when a selector changes,
    # so the dependent dropdowns otherwise keep showing stale options.
    c1, c2 = st.columns(2)
    with c1:
        patient_name = st.text_input("Patient name *")
        age = st.number_input("Age", min_value=0, max_value=120, value=25)
        phone = st.text_input("Phone number (optional)")
        hospital_options = sorted([h for h in doctors_df["hospital_name"].dropna().astype(str).unique().tolist() if h.strip()])
        hospital_name = st.selectbox("Choose hospital (Ranchi)", hospital_options)
        hospital_doctors = doctors_df[
            doctors_df["hospital_name"].astype(str) == hospital_name
        ]
        # Use an explicit hospital-to-department directory so the list is
        # complete even when an older doctors.csv has incomplete entries.
        HOSPITAL_DEPARTMENTS = {
            "Paras HEC Hospital": ["General Medicine", "Oncology", "Orthopedics", "Urology"],
            "Orchid Medical Centre": ["Cardiology"],
            "Santevita Hospital": ["General Medicine", "General Surgery"],
        }
        hospital_departments = HOSPITAL_DEPARTMENTS.get(
            hospital_name,
            sorted(hospital_doctors["department"].dropna().astype(str).unique().tolist())
        )
        department = st.selectbox("Choose department", hospital_departments)
        doctor_options = hospital_doctors[
            hospital_doctors["department"].astype(str) == department
        ]["doctor_name"].astype(str).tolist()
        selected_doctor_name = st.selectbox(
            "Choose doctor",
            ["Auto-assign next available"] + doctor_options
        )
    with c2:
        symptoms = st.text_area("Reason for visit / symptoms (optional)")
        priority = st.selectbox("Queue category entered by staff", PRIORITIES)
        st.caption("Priority is a staff-entered administrative category, not an AI diagnosis.")
    register = st.button("Register Patient", use_container_width=True)

    if register:
        if not patient_name.strip():
            st.error("Please enter the patient's name.")
        else:
            token = next_token(patients)
            new_patient = pd.DataFrame([{
                "token": token,
                "patient_name": patient_name.strip(),
                "age": int(age),
                "phone": phone.strip(),
                "department": department,
                "symptoms": symptoms.strip(),
                "priority": priority,
                "status": "Waiting",
                "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "assigned_doctor_id": "",
                "hospital_name": hospital_name
            }])
            patients = pd.concat([patients, new_patient], ignore_index=True)
            # Keep the assignment column as text so IDs such as D001 are safe.
            patients["assigned_doctor_id"] = (
                patients["assigned_doctor_id"].fillna("").astype(str)
            )
            new_patient_index = patients.index[-1]
            if selected_doctor_name != "Auto-assign next available":
                preferred = hospital_doctors[
                    (hospital_doctors["doctor_name"].astype(str) == selected_doctor_name)
                    & (hospital_doctors["availability"].astype(str) == "Available")
                ]
                reserved = patients[
                    (patients.index != new_patient_index)
                    & (patients["assigned_doctor_id"].astype(str) == str(preferred.iloc[0]["doctor_id"]))
                    & (patients["status"].isin(["Waiting", "In Consultation"]))
                ] if not preferred.empty else pd.DataFrame()
                if not preferred.empty and reserved.empty:
                    assigned_doctor = str(preferred.iloc[0]["doctor_id"])
                    patients.loc[new_patient_index, "assigned_doctor_id"] = assigned_doctor
                else:
                    assigned_doctor = assign_doctor_at_registration(
                        patients, doctors_df, new_patient_index
                    )
            else:
                assigned_doctor = assign_doctor_at_registration(
                    patients, doctors_df, new_patient_index
                )
            save_df(patients, PATIENTS_FILE)
            if assigned_doctor:
                doctor_name_rows = doctors_df[
                    doctors_df["doctor_id"].astype(str) == assigned_doctor
                ]
                doctor_name = (
                    str(doctor_name_rows.iloc[0]["doctor_name"])
                    if not doctor_name_rows.empty else assigned_doctor
                )
                st.success(
                    f"Patient registered successfully. Token: {token}. "
                    f"Assigned doctor: {doctor_name} ({assigned_doctor})."
                )
                st.info(
                    "The doctor is reserved for this waiting patient and remains Available "
                    "until the consultation starts. Queue Management will mark the doctor Busy "
                    "when the patient moves to In Consultation."
                )
            else:
                st.success(f"Patient registered successfully. Token number: {token}")
                st.warning(
                    f"No unreserved Available doctor was found in {department}. "
                    "The patient remains in the queue without an assigned doctor. "
                    "Staff can assign a doctor when one becomes available."
                )
            st.info(priority_note(priority))

    st.markdown("#### Recently registered patients")
    if not patients.empty:
        st.dataframe(patients.tail(10).iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.info("No patients registered yet.")


# ------------------------------------------------------------
# 2. Hospital Queue Management
# ------------------------------------------------------------
elif page == "Queue Management":
    st.header("2. Hospital Queue Management")
    st.write("View department queues and update patient progress.")

    if patients.empty:
        st.info("No patients are registered yet. Start with Patient Registration.")
    else:
        department_filter = st.selectbox("Select department", ["All departments"] + DEPARTMENTS)
        status_filter = st.multiselect(
            "Show status",
            ["Waiting", "In Consultation", "Completed", "Cancelled"],
            default=["Waiting", "In Consultation"]
        )
        queue_view = patients.copy()
        if department_filter != "All departments":
            queue_view = queue_view[queue_view["department"] == department_filter]
        if status_filter:
            queue_view = queue_view[queue_view["status"].isin(status_filter)]
        # Show higher staff-entered priority categories first, then registration order.
        priority_order = {"Emergency": 0, "High": 1, "Moderate": 2, "Normal": 3}
        queue_view["_priority_order"] = queue_view["priority"].map(priority_order).fillna(4)
        queue_view = queue_view.sort_values(["_priority_order", "registered_at"])
        st.dataframe(
            queue_view.drop(columns=["_priority_order"]),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("#### Update patient queue status")
        token_options = patients["token"].dropna().astype(str).tolist()
        if token_options:
            selected_token = st.selectbox("Patient token", token_options)
            new_status = st.selectbox(
                "New status",
                ["Waiting", "In Consultation", "Completed", "Cancelled"]
            )
            if st.button("Update Queue Status"):
                patient_index = patients.index[patients["token"].astype(str) == selected_token][0]
                old_status = patients.loc[patient_index, "status"]
                if old_status == new_status:
                    st.info(f"Token {selected_token} is already {new_status}.")
                else:
                    updated_doctors, assigned_doctor, error = update_doctor_for_queue_change(
                        patients, doctors_df, patient_index, new_status
                    )
                    if error:
                        st.error(error)
                    else:
                        patients.loc[patient_index, "status"] = new_status
                        newly_assigned = []
                        if old_status == "In Consultation" and new_status != "In Consultation":
                            released_department = str(patients.loc[patient_index, "department"])
                            patients, updated_doctors, newly_assigned = assign_waiting_patients_for_department(
                                patients, updated_doctors, released_department
                            )
                        save_df(patients, PATIENTS_FILE)
                        save_df(updated_doctors, DOCTORS_FILE)
                        if new_status == "In Consultation" and assigned_doctor:
                            st.success(
                                f"Token {selected_token} updated. Doctor {assigned_doctor} is now Busy."
                            )
                        elif old_status == "In Consultation" and new_status != "In Consultation":
                            release_message = (
                                f"Token {selected_token} updated. Its assigned doctor was released "
                                "when no other consultation was active."
                            )
                            if newly_assigned:
                                release_message += " Next waiting patient reservation: " + ", ".join(
                                    f"{token} → {doctor_id}" for token, doctor_id in newly_assigned
                                )
                            st.success(release_message)
                        else:
                            st.success(f"Token {selected_token} updated to {new_status}.")
                        st.rerun()

        st.divider()
        st.markdown("#### Admin: Remove a patient record")
        st.warning(
            "This permanently removes the selected patient from the local patient CSV. "
            "Use only for test/demo records or when an authorized administrator approves removal."
        )
        delete_tokens = patients["token"].dropna().astype(str).tolist()
        if delete_tokens:
            delete_token = st.selectbox(
                "Patient token to remove",
                delete_tokens,
                key="delete_patient_token"
            )
            delete_confirm = st.checkbox(
                f"I confirm that patient {delete_token} should be permanently removed.",
                key="confirm_delete_patient"
            )
            if st.button(
                "Delete Patient Record",
                type="primary",
                disabled=not delete_confirm,
                key="delete_patient_record"
            ):
                delete_index = patients.index[
                    patients["token"].astype(str) == delete_token
                ][0]
                delete_status = str(patients.loc[delete_index, "status"])
                updated_doctors = doctors_df.copy()

                # If the patient is currently in consultation, release the
                # workflow-assigned doctor safely before removing the record.
                if delete_status == "In Consultation":
                    updated_doctors, _, release_error = update_doctor_for_queue_change(
                        patients, updated_doctors, delete_index, "Cancelled"
                    )
                    if release_error:
                        st.error(
                            f"Patient was not deleted because the assigned doctor "
                            f"could not be safely released: {release_error}"
                        )
                        st.stop()

                deleted_department = str(patients.loc[delete_index, "department"])
                patients = patients.drop(index=delete_index).reset_index(drop=True)
                patients["assigned_doctor_id"] = (
                    patients["assigned_doctor_id"].fillna("").astype(str)
                )
                patients, updated_doctors, newly_assigned = assign_waiting_patients_for_department(
                    patients, updated_doctors, deleted_department
                )
                save_df(patients, PATIENTS_FILE)
                save_df(updated_doctors, DOCTORS_FILE)
                delete_message = (
                    f"Patient {delete_token} was removed. Any workflow-assigned "
                    "doctor was released when appropriate."
                )
                if newly_assigned:
                    delete_message += " Waiting patient reservation(s): " + ", ".join(
                        f"{token} → {doctor_id}" for token, doctor_id in newly_assigned
                    )
                st.success(delete_message)
                st.rerun()

        st.markdown("#### Current queue summary")
        active = patients[patients["status"].isin(["Waiting", "In Consultation"])]
        q1, q2, q3 = st.columns(3)
        q1.metric("Active patients", len(active))
        q2.metric("Waiting", int((active["status"] == "Waiting").sum()))
        q3.metric("In consultation", int((active["status"] == "In Consultation").sum()))


# ------------------------------------------------------------
# 3. AI Waiting-Time Prediction (original feature retained)
# ------------------------------------------------------------
elif page == "AI Waiting-Time Prediction":
    st.header("3. AI-Based Patient Waiting-Time Prediction")
    st.write("Estimate waiting time using department, queue size, doctor count and time details.")

    patients = read_csv_safe(PATIENTS_FILE, PATIENT_COLUMNS)
    doctors_df = read_csv_safe(DOCTORS_FILE, DOCTOR_COLUMNS)

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            department = st.selectbox("Hospital department", DEPARTMENTS)
            patients_ahead = st.number_input(
                "Patients ahead in queue", min_value=0, max_value=100, value=8
            )
        with c2:
            available_count = int(
                ((doctors_df["department"] == department)
                 & (doctors_df["availability"] == "Available")).sum()
            )
            doctors_default = max(1, available_count) if available_count else 2
            doctors = st.number_input(
                "Available doctors", min_value=1, max_value=20,
                value=min(20, doctors_default)
            )
            avg_consult = st.number_input(
                "Average consultation time (minutes)",
                min_value=5, max_value=60, value=15
            )
        with c3:
            hour = st.slider("Current hour (24-hour format)", 8, 20, 11)
            weekend = st.selectbox("Day type", ["Weekday", "Weekend"])
            token = st.text_input("Patient token (optional)")
        submitted = st.form_submit_button(
            "🔍 Predict Waiting Time", use_container_width=True
        )

    if submitted:
        is_weekend = 1 if weekend == "Weekend" else 0
        sample = pd.DataFrame([{
            "department": department,
            "patients_ahead": int(patients_ahead),
            "available_doctors": int(doctors),
            "avg_consultation_min": int(avg_consult),
            "hour": int(hour),
            "is_weekend": is_weekend
        }])
        prediction = max(1, round(float(model.predict(sample)[0])))
        status = "Low" if prediction < 25 else ("Moderate" if prediction < 60 else "High")

        st.divider()
        st.subheader("Prediction result")
        m1, m2, m3 = st.columns(3)
        m1.metric("Estimated waiting time", f"{prediction} min")
        m2.metric("Queue status", status)
        m3.metric("Department", department)

        if status == "Low":
            st.success("Estimated queue waiting time is relatively low.")
        elif status == "Moderate":
            st.warning("Estimated queue waiting time is moderate. Confirm with hospital staff.")
        else:
            st.error("Estimated queue waiting time is high. Please confirm with hospital staff.")

        new_row = pd.DataFrame([{
            "token": token.strip(),
            "department": department,
            "patients_ahead": int(patients_ahead),
            "available_doctors": int(doctors),
            "avg_consultation_min": int(avg_consult),
            "hour": int(hour),
            "is_weekend": is_weekend,
            "priority": "",
            "predicted_wait_min": prediction,
            "queue_status": status,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }])
        history = pd.concat([history, new_row], ignore_index=True)
        save_df(history, HISTORY_FILE)
        st.caption("Prediction saved to local demonstration history.")

    st.info(
        "The model is trained on synthetic sample data. Real hospital use requires "
        "representative approved data, validation and operational review."
    )


# ------------------------------------------------------------
# 4. Doctor Availability
# ------------------------------------------------------------
elif page == "Doctor Availability":
    st.header("4. Doctor Availability")
    st.write("Doctor availability is synchronized automatically when queue status changes to or from In Consultation. Staff can still update breaks and off-duty status manually.")

    st.markdown("#### Current doctor directory")
    if not doctors_df.empty:
        st.dataframe(doctors_df, use_container_width=True, hide_index=True)
        available = int((doctors_df["availability"] == "Available").sum())
        busy = int((doctors_df["availability"] == "Busy").sum())
        unavailable = int((doctors_df["availability"] == "Unavailable").sum())
        d1, d2, d3 = st.columns(3)
        d1.metric("Available", available)
        d2.metric("Busy", busy)
        d3.metric("Unavailable", unavailable)
    else:
        st.info("No doctor records found.")

    st.markdown("#### Update a doctor's availability")
    if not doctors_df.empty:
        doctor_ids = doctors_df["doctor_id"].astype(str).tolist()
        selected_doctor = st.selectbox("Doctor ID", doctor_ids)
        updated_availability = st.selectbox(
            "Availability", ["Available", "Busy", "Unavailable"]
        )
        if st.button("Save Doctor Availability"):
            doctor_row = doctors_df[
                doctors_df["doctor_id"].astype(str) == selected_doctor
            ].iloc[0]
            active_use = patients[
                (patients["assigned_doctor_id"].astype(str) == str(selected_doctor))
                & (patients["status"].astype(str) == "In Consultation")
            ] if not patients.empty else pd.DataFrame()

            if updated_availability == "Available" and not active_use.empty:
                st.error(
                    "This doctor has an active consultation. Complete or cancel that "
                    "consultation before marking the doctor Available."
                )
            else:
                doctors_df.loc[
                    doctors_df["doctor_id"].astype(str) == selected_doctor,
                    "availability"
                ] = updated_availability

                newly_assigned = []
                if updated_availability == "Available":
                    patients, doctors_df, newly_assigned = assign_waiting_patients_for_department(
                        patients, doctors_df, doctor_row["department"]
                    )
                    save_df(patients, PATIENTS_FILE)

                save_df(doctors_df, DOCTORS_FILE)
                if newly_assigned:
                    st.success(
                        "Doctor availability updated. Waiting patient(s) automatically "
                        "reserved: " + ", ".join(
                            f"{token} → {doctor_id}" for token, doctor_id in newly_assigned
                        )
                    )
                else:
                    st.success("Doctor availability updated.")
                st.rerun()

    st.caption("Automatic workflow: Waiting → In Consultation assigns an Available doctor in the same department and marks that doctor Busy. Completed/Cancelled releases the assigned doctor when no other consultation is active. Breaks and off-duty changes remain staff-controlled.")
    st.caption("The initial doctor directory is sample data. Replace it with authorized, current hospital information for a real deployment.")


# ------------------------------------------------------------
# 5. Patient Priority Classification
# ------------------------------------------------------------
elif page == "Patient Priority Classification":
    st.header("5. Patient Priority Classification")
    st.write(
        "Combined rule-based and AI-assisted demonstration suggestions. "
        "Suggestions are administrative aids only; authorized staff must review and approve the final category."
    )

    if patients.empty:
        st.info("Register a patient first to assign or review a queue category.")
    else:
        token_list = patients["token"].dropna().astype(str).tolist()
        selected_token = st.selectbox("Select patient token", token_list)
        patient_index = patients.index[patients["token"].astype(str) == selected_token][0]
        patient_row = patients.loc[patient_index]
        symptoms_text = str(patient_row.get("symptoms", "")).lower()

        # Transparent keyword rules for an educational demonstration only.
        emergency_terms = [
            "chest pain", "difficulty breathing", "trouble breathing",
            "cannot breathe", "unconscious", "not responding",
            "severe bleeding", "heavy bleeding", "seizure"
        ]
        high_terms = [
            "severe pain", "high fever", "persistent vomiting",
            "fainting", "worsening pain"
        ]
        moderate_terms = [
            "fever", "headache", "vomiting", "pain", "cough",
            "dizziness", "rash"
        ]

        matched_emergency = [term for term in emergency_terms if term in symptoms_text]
        matched_high = [term for term in high_terms if term in symptoms_text]
        matched_moderate = [term for term in moderate_terms if term in symptoms_text]

        if matched_emergency:
            rule_suggestion = "Emergency"
            rule_reason = "A safety keyword was found. Alert qualified hospital staff immediately."
        elif matched_high:
            rule_suggestion = "High"
            rule_reason = "A keyword requiring prompt staff review was found."
        elif matched_moderate:
            rule_suggestion = "Moderate"
            rule_reason = "A symptom keyword was found; staff should review the patient."
        else:
            rule_suggestion = "Normal"
            rule_reason = "No configured keywords matched. This does not mean the patient is medically well."

        # AI-assisted pattern output is displayed separately for learning.
        # It is not a trained or clinically validated machine-learning model.
        if matched_emergency:
            ai_suggestion = "Emergency"
            ai_reason = "Pattern matcher detected an emergency-warning phrase."
        elif matched_high:
            ai_suggestion = "High"
            ai_reason = "Pattern matcher detected a prompt-review phrase."
        elif matched_moderate:
            ai_suggestion = "Moderate"
            ai_reason = "Pattern matcher detected a symptom phrase."
        else:
            ai_suggestion = "Normal"
            ai_reason = "No configured symptom pattern was detected."

        priority_rank = {"Normal": 0, "Moderate": 1, "High": 2, "Emergency": 3}
        combined_suggestion = max(
            [rule_suggestion, ai_suggestion],
            key=lambda item: priority_rank[item]
        )

        st.markdown("#### Patient details")
        st.write(
            f"**Patient:** {patient_row['patient_name']}  |  "
            f"**Token:** {selected_token}  |  "
            f"**Department:** {patient_row['department']}"
        )
        st.write(f"**Recorded reason for visit:** {patient_row.get('symptoms', '') or 'Not provided'}")
        st.write(f"**Current staff-approved category:** {patient_row['priority']}")

        st.markdown("#### Automatic suggestions")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rule-based suggestion", rule_suggestion)
        c2.metric("AI-assisted pattern suggestion", ai_suggestion)
        c3.metric("Combined suggestion", combined_suggestion)
        st.caption(f"Rule explanation: {rule_reason}")
        st.caption(f"AI-assisted explanation: {ai_reason}")

        if matched_emergency:
            st.error(
                "SAFETY ALERT: A configured emergency-warning phrase was detected. "
                "Alert qualified hospital staff immediately and follow the hospital's emergency protocol. "
                "Do not wait for or rely on this software."
            )
        else:
            st.info(
                "These keyword-based demonstration suggestions are not medical triage, "
                "do not assess clinical urgency, and may miss serious conditions."
            )

        st.markdown("#### Staff review and approval")
        approved_priority = st.selectbox(
            "Final queue category (authorized staff must choose)",
            PRIORITIES,
            index=PRIORITIES.index(combined_suggestion),
            key=f"approved_priority_{selected_token}"
        )
        if st.button("Approve and Update Patient Category"):
            patients.loc[patient_index, "priority"] = approved_priority
            save_df(patients, PATIENTS_FILE)
            st.success(f"Staff-approved queue category saved as {approved_priority}.")
            st.info(priority_note(approved_priority))
            st.rerun()

        st.markdown("#### Category guidance")
        for level in PRIORITIES:
            st.write(f"**{level}:** {priority_note(level)}")

    st.warning(
        "Important: This is an educational keyword-matching prototype, not a validated AI clinical model "
        "or emergency triage system. It must never automatically diagnose, downgrade urgency, or delay care. "
        "Qualified staff must assess patients and approve all queue categories."
    )


# ------------------------------------------------------------
# 6. Admin Dashboard (original analytics retained and expanded)
# ------------------------------------------------------------
elif page == "Admin Dashboard":
    st.header("6. Admin Dashboard")
    st.caption("Live operational snapshot from locally stored demonstration records.")
    st.markdown("<div class=\"section-tag\">LIVE PERFORMANCE OVERVIEW</div>", unsafe_allow_html=True)

    patients = read_csv_safe(PATIENTS_FILE, PATIENT_COLUMNS)
    doctors_df = read_csv_safe(DOCTORS_FILE, DOCTOR_COLUMNS)
    history = read_csv_safe(HISTORY_FILE, HISTORY_COLUMNS)

    active_patients = patients[patients["status"].isin(["Waiting", "In Consultation"])] if not patients.empty else patients
    available_doctors = int((doctors_df["availability"] == "Available").sum()) if not doctors_df.empty else 0

    a, b, c, d = st.columns(4)
    a.metric("Registered patients", len(patients))
    b.metric("Active queue", len(active_patients))
    c.metric("Available doctors", available_doctors)
    d.metric("Total predictions", len(history))

    st.markdown("#### 📈 Department & patient insights")

    px.defaults.template = "plotly_white"
    px.defaults.color_discrete_sequence = ["#0F9D91", "#244B69", "#78B7C5", "#F2B66D", "#8297AA"]
    left, right = st.columns(2)
    with left:
        if not patients.empty:
            dept_counts = patients.groupby("department", as_index=False).size()
            st.plotly_chart(
                px.bar(
                    dept_counts, x="department", y="size",
                    title="Registered patients by department",
                    labels={"size": "Patients", "department": "Department"}
                ),
                use_container_width=True
            )
        else:
            st.info("Patient registration chart will appear after patients are registered.")

    with right:
        if not patients.empty:
            priority_counts = patients.groupby("priority", as_index=False).size()
            st.plotly_chart(
                px.pie(
                    priority_counts, names="priority", values="size",
                    title="Patient queue categories"
                ),
                use_container_width=True
            )
        else:
            st.info("Priority chart will appear after patients are registered.")

    if not history.empty:
        st.plotly_chart(
            px.histogram(
                history, x="predicted_wait_min", nbins=12,
                title="Distribution of predicted waiting times",
                labels={"predicted_wait_min": "Minutes"}
            ),
            use_container_width=True
        )
        dept_wait = history.groupby("department", as_index=False)["predicted_wait_min"].mean()
        st.plotly_chart(
            px.bar(
                dept_wait, x="department", y="predicted_wait_min",
                title="Average predicted wait by department",
                labels={"predicted_wait_min": "Minutes", "department": "Department"}
            ),
            use_container_width=True
        )
        st.markdown("#### Recent prediction history")
        st.dataframe(history.tail(15).iloc[::-1], use_container_width=True, hide_index=True)
        st.download_button(
            "Download prediction history (CSV)",
            history.to_csv(index=False).encode("utf-8"),
            "prediction_history.csv",
            "text/csv"
        )
    else:
        st.info("Prediction charts will appear after you make predictions.")

    if not patients.empty:
        st.markdown("#### Recent patient registrations")
        st.dataframe(patients.tail(10).iloc[::-1], use_container_width=True, hide_index=True)
        st.download_button(
            "Download patient register (CSV)",
            patients.to_csv(index=False).encode("utf-8"),
            "patients.csv",
            "text/csv"
        )


# ------------------------------------------------------------
# About
# ------------------------------------------------------------
elif page == "About Project":
    st.header("About Smart Hospital")
    st.write("""
    **Problem:** Patients may not know how long they could wait before a consultation.

    **Solution:** This prototype combines patient registration, queue management,
    doctor availability, staff-entered queue categories, AI waiting-time estimation
    and an administrative dashboard.

    **Machine Learning:** A Random Forest Regression model estimates a continuous
    waiting-time value from department, patients ahead, available doctors,
    average consultation duration, hour and day type.

    **Data storage:** Patient records, doctor availability and prediction history
    are stored in local CSV files under the `data` folder.

    **Limitations:** Model training uses synthetic data. The system is an educational
    prototype, not a clinical decision-support or emergency triage system. Real
    deployment requires approved data, privacy/security controls, validation and
    review by hospital operations and clinical staff.
    """)
    st.markdown("#### Six project features")
    st.markdown("""
    1. Patient Registration
    2. Hospital Queue Management
    3. AI-Based Waiting-Time Prediction
    4. Doctor Availability
    5. Patient Priority Classification (staff-entered category)
    6. Admin Dashboard with charts and reports
    """)
    st.markdown("#### Model workflow")
    st.code(
        "Patient / Queue Details → Preprocessing → Random Forest Regression "
        "→ Estimated Waiting Time → History → Admin Dashboard",
        language="text"
    )