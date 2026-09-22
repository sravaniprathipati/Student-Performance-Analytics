import pandas as pd

# -----------------------------------
# 1. STANDARDIZE COLUMN NAMES
# -----------------------------------
def standardize_columns(df):
    df = df.copy()

    df.columns = [col.strip().lower() for col in df.columns]

    column_map = {
        'cat1': 'mid1',
        'mid 1': 'mid1',
        'mid1': 'mid1',

        'cat2': 'mid2',
        'mid 2': 'mid2',
        'mid2': 'mid2',

        'fat': 'final_exam',
        'final': 'final_exam',
        'final exam': 'final_exam',
        'endsem': 'final_exam',

        'quiz1': 'quiz1',
        'quiz 1': 'quiz1',
        'quiz2': 'quiz2',
        'quiz 2': 'quiz2',
        'quiz3': 'quiz3',
        'quiz 3': 'quiz3',

        'attendance': 'attendance',
        'attendance percentage': 'attendance',
        'attended classes': 'attended',
        'total classes': 'total'
    }

    df.rename(columns=column_map, inplace=True)

    return df


# -----------------------------------
# 2. COMPUTE ATTENDANCE
# -----------------------------------
def compute_attendance(df):
    df = df.copy()

    if 'attended' in df.columns and 'total' in df.columns:
        df['attendance'] = (df['attended'] / df['total']) * 100

    return df


# -----------------------------------
# 3. HANDLE MISSING VALUES
# -----------------------------------
def handle_missing(df):
    df = df.copy()

    defaults = {
        'mid1': 0,
        'mid2': 0,
        'final_exam': 0,
        'quiz1': 0,
        'quiz2': 0,
        'quiz3': 0,
        'attendance': 75
    }

    for col, val in defaults.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(val)

    return df


# -----------------------------------
# 4. CREATE TARGET LABEL (MID2 GRADE)
# -----------------------------------
def create_mid2_grade(df):
    df = df.copy()

    if 'mid2' in df.columns:
        def grade(score):
            if score >= 40: return 'A'
            elif score >= 35: return 'B'
            elif score >= 25: return 'C'
            elif score >= 20: return 'D'
            else: return 'F'

        df['mid2_grade'] = df['mid2'].apply(grade)

    return df


# -----------------------------------
# 5. FINAL PIPELINE (FIXED & SAFE)
# -----------------------------------
def preprocess_pipeline(df):
    df = standardize_columns(df)
    df = compute_attendance(df)
    df = handle_missing(df)
    df = create_mid2_grade(df)

    # 🔥 SAFE NUMERIC CONVERSION (FIX FOR YOUR ERROR)
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    return df