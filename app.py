import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


class FraudDetectionApp:
    def __init__(self):
        self.model = joblib.load("models/fraud_model.pkl")
        self.scaler = joblib.load("models/scaler.pkl")
        self.data = pd.read_csv("dataset/sample_creditcard_10000.csv")
        self.columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    def preprocess(self, df):
        df = df.copy()
        df[["Amount", "Time"]] = self.scaler.transform(df[["Amount", "Time"]])
        return df

    def predict(self, df):
        processed = self.preprocess(df)
        prediction = self.model.predict(processed)
        probability = self.model.predict_proba(processed)[:, 1]
        return prediction, probability

    def dataset_overview(self):
        total = len(self.data)
        fraud = self.data[self.data["Class"] == 1].shape[0]
        normal = self.data[self.data["Class"] == 0].shape[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Transactions", total)
        c2.metric("Legitimate Transactions", normal)
        c3.metric("Fraud Transactions", fraud)

    def prediction_page(self):
        st.subheader("Real-Time Fraud Prediction")

        option = st.selectbox(
            "Choose Input Method",
            ["Real Normal Transaction", "Real Fraud Transaction", "Custom Input"]
        )

        if option == "Real Normal Transaction":
            row = self.data[self.data["Class"] == 0].sample(1).iloc[0]
        elif option == "Real Fraud Transaction":
            row = self.data[self.data["Class"] == 1].sample(1).iloc[0]
        else:
            row = None

        input_values = {}

        col1, col2 = st.columns([1, 2])

        with col1:
            st.write("### Transaction Details")

            input_values["Time"] = st.number_input(
                "Time",
                value=float(row["Time"]) if row is not None else 0.0
            )

            input_values["Amount"] = st.number_input(
                "Amount",
                value=float(row["Amount"]) if row is not None else 100.0
            )

            for i in range(1, 29):
                col = f"V{i}"
                input_values[col] = st.number_input(
                    col,
                    value=float(row[col]) if row is not None else 0.0
                )

        input_df = pd.DataFrame(
            [[input_values[col] for col in self.columns]],
            columns=self.columns
        )

        with col2:
            st.write("### Prediction Output")

            if st.button("Predict Transaction"):
                prediction, probability = self.predict(input_df)

                actual_label = "Custom Input"
                if row is not None:
                    actual_label = "Fraud" if int(row["Class"]) == 1 else "Legitimate"

                m1, m2, m3 = st.columns(3)
                m1.metric("Fraud Probability", f"{probability[0] * 100:.2f}%")
                m2.metric(
                    "Model Prediction",
                    "Fraud" if prediction[0] == 1 else "Legitimate"
                )
                m3.metric("Actual Dataset Label", actual_label)

                if prediction[0] == 1:
                    st.error("⚠ Fraudulent Transaction Detected")
                    st.warning("Action Suggested: Block transaction and alert bank security team.")
                else:
                    st.success("✅ Legitimate Transaction")
                    st.info("Action Suggested: Transaction can be approved.")

                st.write("### Input Transaction")
                st.dataframe(input_df)

    def csv_upload_page(self):
        st.subheader("Batch Fraud Screening")

        uploaded_file = st.file_uploader(
            "Upload CSV file with same dataset columns",
            type=["csv"]
        )

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            if "Class" in df.columns:
                df = df.drop("Class", axis=1)

            if all(col in df.columns for col in self.columns):
                predictions, probabilities = self.predict(df[self.columns])

                result_df = df.copy()
                result_df["Fraud Probability"] = probabilities
                result_df["Prediction"] = [
                    "Fraud" if p == 1 else "Legitimate" for p in predictions
                ]

                st.success("Batch prediction completed successfully.")
                st.dataframe(result_df.head(20))

                fraud_count = result_df[result_df["Prediction"] == "Fraud"].shape[0]
                normal_count = result_df[result_df["Prediction"] == "Legitimate"].shape[0]

                fig, ax = plt.subplots(figsize=(4, 3))
                ax.bar(["Legitimate", "Fraud"], [normal_count, fraud_count])
                ax.set_title("Batch Prediction Result")
                ax.set_ylabel("Count")
                st.pyplot(fig)
            else:
                st.error("CSV columns do not match required dataset format.")

    def evaluation_page(self):
        st.subheader("Dynamic Model Evaluation")

        X = self.data.drop("Class", axis=1)
        y = self.data["Class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        X_test_scaled = self.preprocess(X_test)

        y_pred = self.model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred) * 100
        recall = recall_score(y_test, y_pred) * 100
        f1 = f1_score(y_test, y_pred) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{accuracy:.2f}%")
        c2.metric("Precision", f"{precision:.2f}%")
        c3.metric("Recall", f"{recall:.2f}%")
        c4.metric("F1 Score", f"{f1:.2f}%")

        graph_col1, graph_col2 = st.columns(2)

        with graph_col1:
            metrics = {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1
            }

            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(metrics.keys(), metrics.values())
            ax.set_ylim(0, 100)
            ax.set_title("Model Performance")
            ax.set_ylabel("Score (%)")
            st.pyplot(fig)

        with graph_col2:
            cm = confusion_matrix(y_test, y_pred)

            fig2, ax2 = plt.subplots(figsize=(4, 3))
            ax2.imshow(cm)
            ax2.set_title("Confusion Matrix")
            ax2.set_xlabel("Predicted")
            ax2.set_ylabel("Actual")

            ax2.set_xticks([0, 1])
            ax2.set_yticks([0, 1])
            ax2.set_xticklabels(["Legitimate", "Fraud"])
            ax2.set_yticklabels(["Legitimate", "Fraud"])

            for i in range(2):
                for j in range(2):
                    ax2.text(j, i, cm[i, j], ha="center", va="center")

            st.pyplot(fig2)

        st.write("### Evaluation Explanation")
        st.info(
            "These metrics are calculated dynamically using the saved trained model "
            "on the test split of the dataset."
        )

    def run(self):
        st.set_page_config(
            page_title="Fraud Detection System",
            page_icon="💳",
            layout="wide"
        )

        st.title("💳 AI-Based Credit Card Fraud Detection System")
        st.write(
            "A machine learning dashboard for real-time and batch credit card fraud detection."
        )

        self.dataset_overview()

        tab1, tab2, tab3 = st.tabs([
            "Prediction",
            "Batch Screening",
            "Model Evaluation"
        ])

        with tab1:
            self.prediction_page()

        with tab2:
            self.csv_upload_page()

        with tab3:
            self.evaluation_page()


app = FraudDetectionApp()
app.run()