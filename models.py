
import pandas as pd
import xgboost as xgb
import numpy as np

from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

from utils import calculate_lci_stage1


class StudentRiskPredictor:

    def __init__(self):
        self.model_cat2_reg = None
        self.model_fat_reg = None
        self.model_cat2_clf = None
        self.model_fat_clf = None

        self.cat2_label_encoder = LabelEncoder()
        self.fat_label_encoder = LabelEncoder()

        self.best_cat2_model_type = "regressor"
        self.best_fat_model_type = "regressor"

        self.metrics = {}

    # -------------------------------
    # LABEL FUNCTIONS
    # -------------------------------
    def mid2_risk_label(self, x):
        if x >= 30:
            return "LOW"
        elif x >= 20:
            return "MODERATE"
        else:
            return "HIGH"

    def final_risk_label(self, x):
        if x >= 60:
            return "LOW"
        elif x >= 45:
            return "MODERATE"
        else:
            return "HIGH"

    # -------------------------------
    # STAGE 1 FEATURES
    # -------------------------------
    def engineer_features_stage1(self, cat1, quiz1):
        cat1_n = cat1 / 50
        quiz1_n = quiz1 / 20

        return pd.DataFrame({
            "mid1": [cat1],
            "quiz1": [quiz1],
            "mid1_n": [cat1_n],
            "quiz1_n": [quiz1_n],
            "lci_stage1": [
                calculate_lci_stage1(
                    pd.Series([cat1]),
                    pd.Series([quiz1])
                ).values[0]
            ],
            "weighted_stage1": [(0.70 * cat1_n) + (0.30 * quiz1_n)],
            "gap_mid1_quiz1": [cat1_n - quiz1_n],
            "low_mid1_flag": [1 if cat1 < 25 else 0],
            "safe_mid1_flag": [1 if cat1 >= 30 else 0],
            "high_quiz1_flag": [1 if quiz1 >= 16 else 0],
        })

    def _build_stage1_matrix(self, df):
        mid1_n = df["mid1"] / 50
        quiz1_n = df["quiz1"] / 20

        return pd.DataFrame({
            "mid1": df["mid1"],
            "quiz1": df["quiz1"],
            "mid1_n": mid1_n,
            "quiz1_n": quiz1_n,
            "lci_stage1": calculate_lci_stage1(df["mid1"], df["quiz1"]),
            "weighted_stage1": (0.70 * mid1_n) + (0.30 * quiz1_n),
            "gap_mid1_quiz1": mid1_n - quiz1_n,
            "low_mid1_flag": (df["mid1"] < 25).astype(int),
            "safe_mid1_flag": (df["mid1"] >= 30).astype(int),
            "high_quiz1_flag": (df["quiz1"] >= 16).astype(int),
        })

    # -------------------------------
    # STAGE 2 FEATURES
    # -------------------------------
    def engineer_features_stage2(self, mid2, quiz2, quiz3, attendance):
        mid2_n = mid2 / 50
        q2_n = quiz2 / 20
        q3_n = quiz3 / 20
        att_n = attendance / 100
        quiz_avg = (q2_n + q3_n) / 2

        return pd.DataFrame({
            "mid2": [mid2],
            "quiz2": [quiz2],
            "quiz3": [quiz3],
            "attendance": [attendance],
            "mid2_n": [mid2_n],
            "quiz2_n": [q2_n],
            "quiz3_n": [q3_n],
            "attendance_n": [att_n],
            "quiz_avg": [quiz_avg],
            "quiz_gap": [abs(q2_n - q3_n)],
            "mid2_attendance_interaction": [mid2_n * att_n],
            "quiz_attendance_interaction": [quiz_avg * att_n],
            "low_mid2_flag": [1 if mid2 < 20 else 0],
            "safe_mid2_flag": [1 if mid2 >= 30 else 0],
            "low_attendance_flag": [1 if attendance < 60 else 0],
            "very_low_attendance_flag": [1 if attendance < 40 else 0],
        })

    def _build_stage2_matrix(self, df):
        mid2_n = df["mid2"] / 50
        q2_n = df["quiz2"] / 20
        q3_n = df["quiz3"] / 20
        att_n = df["attendance"] / 100
        quiz_avg = (q2_n + q3_n) / 2

        return pd.DataFrame({
            "mid2": df["mid2"],
            "quiz2": df["quiz2"],
            "quiz3": df["quiz3"],
            "attendance": df["attendance"],
            "mid2_n": mid2_n,
            "quiz2_n": q2_n,
            "quiz3_n": q3_n,
            "attendance_n": att_n,
            "quiz_avg": quiz_avg,
            "quiz_gap": abs(q2_n - q3_n),
            "mid2_attendance_interaction": mid2_n * att_n,
            "quiz_attendance_interaction": quiz_avg * att_n,
            "low_mid2_flag": (df["mid2"] < 20).astype(int),
            "safe_mid2_flag": (df["mid2"] >= 30).astype(int),
            "low_attendance_flag": (df["attendance"] < 60).astype(int),
            "very_low_attendance_flag": (df["attendance"] < 40).astype(int),
        })

    # -------------------------------
    # MODEL BUILDERS
    # -------------------------------
    def _make_regressor(self):
        return xgb.XGBRegressor(
            n_estimators=180,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.5,
            reg_alpha=0.4,
            objective="reg:squarederror",
            random_state=42
        )

    def _make_classifier(self):
        return xgb.XGBClassifier(
            n_estimators=180,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.5,
            reg_alpha=0.4,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42
        )

    def _make_calibrated_classifier(self, y_labels):
        base_clf = self._make_classifier()
        min_class_count = y_labels.value_counts().min()

        if min_class_count < 3:
            return base_clf

        try:
            return CalibratedClassifierCV(
                estimator=base_clf,
                method="sigmoid",
                cv=3
            )
        except TypeError:
            return CalibratedClassifierCV(
                base_estimator=base_clf,
                method="sigmoid",
                cv=3
            )

    # -------------------------------
    # CROSS VALIDATION
    # -------------------------------
    def _cv_compare_models(self, X, y_marks, y_labels, label_func, max_score):
        class_counts = y_labels.value_counts()
        min_class_count = class_counts.min()

        if min_class_count < 2:
            return 0.0, 0.0, 0.0, 0.0, "regressor"

        n_splits = min(5, int(min_class_count))

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )

        reg_f1_scores = []
        clf_f1_scores = []

        for train_idx, test_idx in skf.split(X, y_labels):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train_marks = y_marks.iloc[train_idx]
            y_test_marks = y_marks.iloc[test_idx]
            y_train_labels = y_labels.iloc[train_idx]
            y_test_labels = y_labels.iloc[test_idx]

            reg = self._make_regressor()
            reg.fit(X_train, y_train_marks)

            pred_marks = reg.predict(X_test)
            pred_marks = np.clip(pred_marks, 0, max_score)
            pred_labels = pd.Series(pred_marks).apply(label_func)

            reg_f1_scores.append(
                f1_score(
                    y_test_labels,
                    pred_labels,
                    average="weighted",
                    zero_division=0
                )
            )

            le = LabelEncoder()
            y_train_encoded = le.fit_transform(y_train_labels)

            clf = self._make_classifier()
            clf.fit(X_train, y_train_encoded)

            pred_encoded = clf.predict(X_test)
            pred_cls_labels = le.inverse_transform(pred_encoded)

            clf_f1_scores.append(
                f1_score(
                    y_test_labels,
                    pred_cls_labels,
                    average="weighted",
                    zero_division=0
                )
            )

        reg_mean = float(np.mean(reg_f1_scores))
        reg_std = float(np.std(reg_f1_scores))
        clf_mean = float(np.mean(clf_f1_scores))
        clf_std = float(np.std(clf_f1_scores))

        best_type = "regressor" if reg_mean >= clf_mean else "classifier"

        return reg_mean, reg_std, clf_mean, clf_std, best_type

    # -------------------------------
    # TRAIN MODELS
    # -------------------------------
    def train_models(self, df):
        df = df.copy()

        required = [
            "mid1", "mid2", "quiz1", "quiz2",
            "quiz3", "attendance", "final_exam"
        ]

        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        df = df.dropna(subset=required)

        df["MID2_RISK"] = df["mid2"].apply(self.mid2_risk_label)
        df["FINAL_RISK"] = df["final_exam"].apply(self.final_risk_label)

        # MID2 models
        X_mid2 = self._build_stage1_matrix(df)
        y_mid2_marks = df["mid2"]
        y_mid2_labels = df["MID2_RISK"]

        reg_mean, reg_std, clf_mean, clf_std, best_type = self._cv_compare_models(
            X_mid2,
            y_mid2_marks,
            y_mid2_labels,
            self.mid2_risk_label,
            max_score=50
        )

        self.best_cat2_model_type = best_type
        self.metrics["mid2_f1"] = max(reg_mean, clf_mean)
        self.metrics["mid2_f1_percent"] = round(self.metrics["mid2_f1"] * 100, 2)
        self.metrics["mid2_f1_cv_std"] = round(
            reg_std if best_type == "regressor" else clf_std,
            4
        )
        self.metrics["mid2_reg_f1"] = reg_mean
        self.metrics["mid2_clf_f1"] = clf_mean

        self.cat2_label_encoder.fit(y_mid2_labels)
        y_mid2_encoded = self.cat2_label_encoder.transform(y_mid2_labels)

        self.model_cat2_reg = self._make_regressor()
        self.model_cat2_reg.fit(X_mid2, y_mid2_marks)

        self.model_cat2_clf = self._make_calibrated_classifier(y_mid2_labels)
        self.model_cat2_clf.fit(X_mid2, y_mid2_encoded)

        # Final exam models
        X_final = self._build_stage2_matrix(df)
        y_final_marks = df["final_exam"]
        y_final_labels = df["FINAL_RISK"]

        reg_mean, reg_std, clf_mean, clf_std, best_type = self._cv_compare_models(
            X_final,
            y_final_marks,
            y_final_labels,
            self.final_risk_label,
            max_score=100
        )

        self.best_fat_model_type = best_type
        self.metrics["final_exam_f1"] = max(reg_mean, clf_mean)
        self.metrics["final_exam_f1_percent"] = round(
            self.metrics["final_exam_f1"] * 100,
            2
        )
        self.metrics["final_exam_f1_cv_std"] = round(
            reg_std if best_type == "regressor" else clf_std,
            4
        )
        self.metrics["final_reg_f1"] = reg_mean
        self.metrics["final_clf_f1"] = clf_mean

        self.fat_label_encoder.fit(y_final_labels)
        y_final_encoded = self.fat_label_encoder.transform(y_final_labels)

        self.model_fat_reg = self._make_regressor()
        self.model_fat_reg.fit(X_final, y_final_marks)

        self.model_fat_clf = self._make_calibrated_classifier(y_final_labels)
        self.model_fat_clf.fit(X_final, y_final_encoded)

        return True

    # -------------------------------
    # METRICS
    # -------------------------------
    def get_main_metrics(self):
        return self.metrics

    # -------------------------------
    # PROBABILITY HELPERS
    # -------------------------------
    def _prob_dict(self, model, encoder, X):
        probs = model.predict_proba(X)[0]
        classes = encoder.inverse_transform(np.arange(len(probs)))

        result = {
            "HIGH": 0.0,
            "MODERATE": 0.0,
            "LOW": 0.0
        }

        for cls, prob in zip(classes, probs):
            result[cls] = float(prob)

        return result

    def _soft_normalize_probs(self, prob_dict):
        total = sum(prob_dict.values())

        if total <= 0:
            return {
                "HIGH": 0.0,
                "MODERATE": 0.0,
                "LOW": 1.0
            }

        return {k: float(v / total) for k, v in prob_dict.items()}

    def _adaptive_fail_probability(self, prob_dict, academic_weakness):
        """
        Adaptive probability:
        - If model confidence is high, trust model more.
        - If model confidence is low, use academic weakness more.
        """
        confidence = abs(prob_dict["HIGH"] - prob_dict["LOW"])
        model_weight = 0.60 + 0.30 * confidence
        academic_weight = 1 - model_weight

        fail_probability = (
            model_weight * prob_dict["HIGH"] +
            academic_weight * academic_weakness
        )

        return float(np.clip(fail_probability, 0.02, 0.95))

    # -------------------------------
    # MID2 PREDICTION
    # -------------------------------
    def predict_cat2_risk(self, cat1, quiz1):
        X = self.engineer_features_stage1(cat1, quiz1)

        predicted_mid2 = float(self.model_cat2_reg.predict(X)[0])
        predicted_mid2 = float(np.clip(predicted_mid2, 0, 50))

        prob_dict = self._prob_dict(
            self.model_cat2_clf,
            self.cat2_label_encoder,
            X
        )

        if cat1 >= 30 and quiz1 >= 18:
            prob_dict["LOW"] *= 1.15
            prob_dict["MODERATE"] *= 0.90
            prob_dict["HIGH"] *= 0.80

        if quiz1 >= 18:
            prob_dict["HIGH"] *= 0.90

        prob_dict = self._soft_normalize_probs(prob_dict)

        if self.best_cat2_model_type == "regressor":
            risk = self.mid2_risk_label(predicted_mid2)
        else:
            risk = max(prob_dict, key=prob_dict.get)

        cat1_strength = cat1 / 50
        quiz1_strength = quiz1 / 20

        academic_weakness = 1 - (
            0.70 * cat1_strength +
            0.30 * quiz1_strength
        )

        fail_probability = self._adaptive_fail_probability(
            prob_dict,
            academic_weakness
        )

        return {
            "prediction": risk,
            "predicted_mid2_score": round(predicted_mid2, 2),
            "probs": {
                "HIGH": prob_dict["HIGH"],
                "MODERATE": prob_dict["MODERATE"],
                "LOW": prob_dict["LOW"]
            },
            "fail_probability": fail_probability,
            "risk_level": risk
        }

    # -------------------------------
    # FINAL EXAM PREDICTION
    # -------------------------------
    def predict_fat_risk(self, cat2_result, quiz2, quiz3, attendance, mid2=None):

        if mid2 is not None and not pd.isna(mid2):
            mid2_score = mid2
        else:
            mid2_score = cat2_result.get("predicted_mid2_score", 25)

        X = self.engineer_features_stage2(
            mid2_score,
            quiz2,
            quiz3,
            attendance
        )

        predicted_final = float(self.model_fat_reg.predict(X)[0])
        predicted_final = float(np.clip(predicted_final, 0, 100))

        prob_dict = self._prob_dict(
            self.model_fat_clf,
            self.fat_label_encoder,
            X
        )

        if attendance < 60:
            prob_dict["HIGH"] *= 1.15
            prob_dict["LOW"] *= 0.90

        if attendance < 40:
            prob_dict["HIGH"] *= 1.35
            prob_dict["LOW"] *= 0.70

        if mid2_score >= 30 and quiz2 >= 15 and quiz3 >= 15 and attendance >= 70:
            prob_dict["LOW"] *= 1.15
            prob_dict["MODERATE"] *= 0.95
            prob_dict["HIGH"] *= 0.85

        prob_dict = self._soft_normalize_probs(prob_dict)

        if self.best_fat_model_type == "regressor":
            risk = self.final_risk_label(predicted_final)
        else:
            risk = max(prob_dict, key=prob_dict.get)

        mid2_strength = mid2_score / 50
        quiz_strength = ((quiz2 / 20) + (quiz3 / 20)) / 2
        attendance_strength = attendance / 100

        academic_weakness = 1 - (
            0.45 * mid2_strength +
            0.30 * quiz_strength +
            0.25 * attendance_strength
        )

        fail_probability = self._adaptive_fail_probability(
            prob_dict,
            academic_weakness
        )

        return {
            "prediction": risk,
            "predicted_final_score": round(predicted_final, 2),
            "probs": {
                "HIGH": prob_dict["HIGH"],
                "MODERATE": prob_dict["MODERATE"],
                "LOW": prob_dict["LOW"]
            },
            "fail_probability": fail_probability,
            "risk_level": risk
        }

    # -------------------------------
    # GLOBAL FEATURE IMPORTANCE
    # -------------------------------
    def get_feature_importance(self, stage="final"):
        if stage == "mid2":
            model = self.model_cat2_reg
            sample_df = pd.DataFrame({
                "mid1": [25],
                "quiz1": [10]
            })
            features = self._build_stage1_matrix(sample_df).columns
        else:
            model = self.model_fat_reg
            sample_df = pd.DataFrame({
                "mid2": [25],
                "quiz2": [10],
                "quiz3": [10],
                "attendance": [70]
            })
            features = self._build_stage2_matrix(sample_df).columns

        importance = model.feature_importances_

        return pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

    # -------------------------------
    # SHAP-STYLE LOCAL EXPLANATION
    # -------------------------------
    def get_local_explanation(self, mid2, quiz2, quiz3, attendance):
        mid2_n = mid2 / 50
        quiz2_n = quiz2 / 20
        quiz3_n = quiz3 / 20
        quiz_avg_n = (quiz2_n + quiz3_n) / 2
        attendance_n = attendance / 100

        local_df = pd.DataFrame({
            "Feature": [
                "Mid 2 Score",
                "Quiz 2 Score",
                "Quiz 3 Score",
                "Quiz Average",
                "Attendance"
            ],
            "Value": [
                f"{mid2}/50",
                f"{quiz2}/20",
                f"{quiz3}/20",
                f"{((quiz2 + quiz3) / 2):.1f}/20",
                f"{attendance}%"
            ],
            "Risk Contribution": [
                round((1 - mid2_n) * 0.35, 3),
                round((1 - quiz2_n) * 0.15, 3),
                round((1 - quiz3_n) * 0.15, 3),
                round((1 - quiz_avg_n) * 0.20, 3),
                round((1 - attendance_n) * 0.35, 3)
            ]
        })

        local_df["Effect"] = local_df["Risk Contribution"].apply(
            lambda x: "Increases Risk" if x > 0.15 else "Low Impact"
        )

        return local_df.sort_values(
            by="Risk Contribution",
            ascending=False
        )

    # -------------------------------
    # RISK DECOMPOSITION
    # -------------------------------
    def get_risk_decomposition(self, cat2_score, quiz2, quiz3, attendance):
        cat2_n = cat2_score / 50
        q_avg = ((quiz2 / 20) + (quiz3 / 20)) / 2
        att_n = attendance / 100

        return {
            "CAT2_Risk": round((1 - cat2_n) * 0.35, 3),
            "Quiz_Risk": round((1 - q_avg) * 0.30, 3),
            "Attendance_Risk": round((1 - att_n) * 0.35, 3)
        }

    # -------------------------------
    # ACTION PLAN
    # -------------------------------
    def generate_action_plan(self, cat2_result, quiz2, quiz3, attendance, mid2=None):

        fat_result = self.predict_fat_risk(
            cat2_result,
            quiz2,
            quiz3,
            attendance,
            mid2=mid2
        )

        current_fail = fat_result.get("fail_probability", 0)

        actions = []

        if current_fail > 0.70:
            target_score = 75
            effort = "VERY HIGH"
        elif current_fail > 0.40:
            target_score = 65
            effort = "HIGH"
        elif current_fail > 0.20:
            target_score = 55
            effort = "MODERATE"
        else:
            target_score = 50
            effort = "LOW"

        actions.append(
            f"🎯 To stay safe, aim for at least {target_score}/100 in final exam ({effort} effort required)"
        )

        quiz_avg = (quiz2 + quiz3) / 2

        if quiz_avg < 10:
            actions.append("📚 Weak foundation → Focus on basic concepts and theory revision")
        elif quiz_avg < 15:
            actions.append("📖 Moderate understanding → Practice previous year questions")
        else:
            actions.append("🧠 Strong understanding → Focus on problem-solving and speed")

        if abs(quiz2 - quiz3) > 5:
            actions.append("⚠️ Performance inconsistency → Revise all units evenly")

        if attendance < 60:
            actions.append("🚨 Low attendance → Increase self-study and cover missed topics")

        if current_fail > 0.70:
            actions.append("🔥 Daily intensive preparation + full syllabus revision required")
        elif current_fail > 0.40:
            actions.append("⚡ Regular study plan + weekly mock tests recommended")
        else:
            actions.append("✅ Maintain performance with revision and practice")

        return actions