import streamlit as st, pandas as pd, joblib, numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

st.set_page_config(page_title="Dropout Risk Predictor", layout="wide")

# Helper Function + Custom Class Preprocessing Pipeline
def cast_to_str(X):
  return X.astype(str)

def cast_to_int(X):
  return X.astype(int)

class IQROutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor=1.5):
      self.factor = factor

    def fit(self, X, y=None):
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        iqr = q3 - q1

        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr

        return self

    def transform(self, X):
      return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None

        if hasattr(input_features, '__len__') and not isinstance(input_features, str):
          return np.asarray(input_features, dtype=object)

        return np.asarray([input_features], dtype=object)

class FeatureNameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        return self.transformer.transform(X)

    def get_feature_names_out(self, input_features=None):
        if hasattr(self.transformer, 'get_feature_names_out'):
            return self.transformer.get_feature_names_out(input_features)
        elif input_features is not None:
            return input_features
        else:
            return None

# Load Saved Model
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

model = load_model("best_dropout_model.joblib")

# Streamlit UI Configuration
st.markdown(
    "<h1 style='text-align: center;'>🎓 Jaya Jaya Institute Dropout Risk Predictor</h1>",
    unsafe_allow_html=True
)
st.write(
    "<h5 style='text-align: center;'>Enter student data and click Predict.</h5>",
    unsafe_allow_html=True
)

# Dictionaries
marital_map = {
    "Single":1, "Married":2, "Widower":3, "Divorced":4, "Facto Union":5, "Legally Separated":6
}

nationality_map = {
    "Portuguese":1, "German":2, "Spanish":6, "Italian":11, "Dutch":13, "English":14,
    "Lithuanian":17, "Angolan":21, "Cape Verdean":22, "Guinean":24, "Mozambican":25,
    "Santomean":26, "Turkish":32, "Brazilian":41, "Romanian":62, "Moldovan":100,
    "Mexican":101, "Ukrainian":103, "Russian":105, "Cuban":108, "Colombian":109
}

prev_qual_map = {
    "Secondary Education":1, "Bachelor's Degree":2, "Higher Degree":3, "Master's":4,
    "Doctorate":5, "Freq Higher Ed":6, "12th Year Not Completed":9, "11th Year Not Completed":10,
    "Other 11th Year":12, "10th Year":14, "10th Not Completed":15, "Basic Ed 3rd Cycle":19,
    "Basic Ed 2nd Cycle":38, "Tech Specialization":39, "Higher Ed 1st Cycle":40,
    "Prof Higher Tech":42, "Higher Ed Master":43
}

occupation_map = {
    0:"Student", 1:"Legislative/Executive Directors", 2:"Specialists in Intellectual Activities",
    3:"Technicians and Professions", 4:"Administrative Staff", 5:"Personal Services & Sellers",
    6:"Agriculture Workers", 7:"Industrial Skilled Workers", 8:"Machine Operators",
    9:"Unskilled Workers", 10:"Armed Forces", 90:"Other", 99:"Unknown"
}
app_mode_map = {
    1:"1st phase - general", 2:"Ordinance 612/93", 5:"1st phase - special Azores",
    7:"Holders of other courses", 10:"Ordinance 854-B/99", 15:"Intl student bachelor",
    16:"1st phase - special Madeira", 17:"2nd phase - general", 18:"3rd phase - general",
    26:"Ordinance 533-A/99 b2", 27:"Ordinance 533-A/99 b3", 39:"Over 23 years old",
    42:"Transfer", 43:"Change of course", 44:"Tech specialization holders",
    51:"Change institution/course", 53:"Short cycle holders", 57:"Change institution (Intl)"
}

app_order_map = {
    i:f"Choice {i+1}" for i in range(10)
}

course_map = {
    33:"Biofuel Tech", 171:"Animation & Multimedia", 8014:"Social Service (Evening)",
    9003:"Agronomy", 9070:"Communication Design", 9085:"Veterinary Nursing",
    9119:"Informatics Engineering", 9130:"Equinculture", 9147:"Management",
    9238:"Social Service", 9254:"Tourism", 9500:"Nursing", 9556:"Oral Hygiene",
    9670:"Advertising & Marketing", 9773:"Journalism & Communication",
    9853:"Basic Education", 9991:"Management (Evening)"
}

# Options lists
marital_opts       = list(marital_map.keys())
nationality_opts   = list(nationality_map.keys())
prev_qual_opts     = list(prev_qual_map.keys())
occupation_opts    = list(occupation_map.keys())
app_mode_opts      = list(app_mode_map.values())
app_order_opts     = list(app_order_map.values())
course_opts        = list(course_map.values())

# Build Input Form
with st.form("input_form"):
    st.header("🧑 Student Personal Information")
    marital      = st.selectbox("Marital Status", marital_opts)
    gender       = st.selectbox("Gender", ["Male","Female"])
    age          = st.number_input("Age at enrollment", 15, 100, 20)
    nationality  = st.selectbox("Nationality", nationality_opts)
    international= st.selectbox("International student?", ["Yes","No"])
    edu_special  = st.selectbox("Educational special needs?", ["Yes","No"])
    displaced    = st.selectbox("Displaced (refugee)?", ["Yes","No"])
    scholarship  = st.selectbox("Scholarship holder?", ["Yes","No"])

    st.header("🎓 Educational Background")
    prev_qual         = st.selectbox("Previous Qualification", prev_qual_opts)
    prev_qual_grade   = st.number_input("Previous qualification grade (0–200)", 0, 200, 100)
    admission_grade   = st.number_input("Admission grade (0–200)", 0, 200, 100)

    st.header("👨‍👩‍👧‍👦 Family Background")
    mom_qual = st.selectbox("Mother's Qualification", prev_qual_opts)
    dad_qual = st.selectbox("Father's Qualification", prev_qual_opts)
    mom_occ  = st.selectbox("Mother's Occupation", occupation_opts)
    dad_occ  = st.selectbox("Father's Occupation", occupation_opts)

    st.header("🏫 Enrollment Information")
    app_mode  = st.selectbox("Application Mode", app_mode_opts)
    app_order = st.selectbox("Application Order", app_order_opts)
    course    = st.selectbox("Course", course_opts)
    attend    = st.selectbox("Attendance time", ["Daytime","Evening"])

    st.header("💰 Administrative Information")
    debtor  = st.selectbox("Debtor?", ["Yes","No"])
    fees_ok = st.selectbox("Tuition fees up to date?", ["Yes","No"])

    st.header("📚 Academic Performance")
    cu1_credited = st.number_input("1st sem credited", 0, 100, 0)
    cu1_enrolled = st.number_input("1st sem enrolled", 0, 100, 0)
    cu1_evals    = st.number_input("1st sem evaluations", 0, 100, 0)
    cu1_approved = st.number_input("1st sem approved", 0, 100, 0)
    cu1_grade    = st.number_input("1st sem grade", 0.0, 20.0, 10.0, 0.1)
    cu1_noeval   = st.number_input("1st sem without evaluations", 0, 100, 0)

    cu2_credited = st.number_input("2nd sem credited", 0, 100, 0)
    cu2_enrolled = st.number_input("2nd sem enrolled", 0, 100, 0)
    cu2_evals    = st.number_input("2nd sem evaluations", 0, 100, 0)
    cu2_approved = st.number_input("2nd sem approved", 0, 100, 0)
    cu2_grade    = st.number_input("2nd sem grade", 0.0, 20.0, 10.0, 0.1)
    cu2_noeval   = st.number_input("2nd sem without evaluations", 0, 100, 0)

    st.header("📈 Economic Conditions at Enrollment")
    unemp = st.number_input("Unemployment rate (%)", 0.0, 100.0, 10.0, 0.1)
    infl  = st.number_input("Inflation rate (%)",   0.0, 100.0, 5.0,  0.1)
    gdp   = st.number_input("GDP", 0.0, None, 10000.0, 100.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    raw = {
        "Marital_status":         marital_map[marital],
        "Gender":                 1 if gender=="Male" else 0,
        "Age_at_enrollment":      age,
        "Nacionality":            nationality_map[nationality],
        "International":          1 if international=="Yes" else 0,
        "Educational_special_needs": 1 if edu_special=="Yes" else 0,
        "Displaced":                1 if displaced=="Yes" else 0,
        "Scholarship_holder":       1 if scholarship=="Yes" else 0,
        "Previous_qualification":    prev_qual_map[prev_qual],
        "Previous_qualification_grade": prev_qual_grade,
        "Admission_grade":            admission_grade,
        "Mothers_qualification":      prev_qual_map[mom_qual],
        "Fathers_qualification":      prev_qual_map[dad_qual],
        "Mothers_occupation":         mom_occ,
        "Fathers_occupation":         dad_occ,
        "Application_mode":           {v:k for k,v in app_mode_map.items()}[app_mode],
        "Application_order":          {v:k for k,v in app_order_map.items()}[app_order],
        "Course":                     {v:k for k,v in course_map.items()}[course],
        "Daytime_evening_attendance": 1 if attend=="Daytime" else 0,
        "Debtor":                     1 if debtor=="Yes" else 0,
        "Tuition_fees_up_to_date":    1 if fees_ok=="Yes" else 0,
        "Curricular_units_1st_sem_credited":             cu1_credited,
        "Curricular_units_1st_sem_enrolled":             cu1_enrolled,
        "Curricular_units_1st_sem_evaluations":          cu1_evals,
        "Curricular_units_1st_sem_approved":             cu1_approved,
        "Curricular_units_1st_sem_grade":                cu1_grade,
        "Curricular_units_1st_sem_without_evaluations":  cu1_noeval,
        "Curricular_units_2nd_sem_credited":             cu2_credited,
        "Curricular_units_2nd_sem_enrolled":             cu2_enrolled,
        "Curricular_units_2nd_sem_evaluations":          cu2_evals,
        "Curricular_units_2nd_sem_approved":             cu2_approved,
        "Curricular_units_2nd_sem_grade":                cu2_grade,
        "Curricular_units_2nd_sem_without_evaluations":  cu2_noeval,
        "Unemployment_rate":  unemp,
        "Inflation_rate":     infl,
        "GDP":                gdp
    }
    
    X_raw = pd.DataFrame([raw])
    proba = model.predict_proba(X_raw)[0,1]
    pred  = model.predict(X_raw)[0]

    st.subheader("📊 Results")
    st.markdown(f"**Probability of Dropout:** {proba:.2%}")
    st.markdown(f"**Prediction:** {'Dropout' if pred==1 else 'Non-Dropout'}")
