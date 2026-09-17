import gradio as gr
from src.serving.inference import make_prediction

def gradio_interface(
    Gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies, Contract,
    PaperlessBilling, PaymentMethod, Tenure, MonthlyCharges, TotalCharges
):
    customer_data = {
        "Gender": Gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "Tenure": Tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
    }
    _, prediction = make_prediction(customer_data)
    return prediction

def build_app(app, path):
    demo = gr.Interface(
        fn=gradio_interface,
        inputs=[
            gr.Dropdown(["Male", "Female"], label="Gender"),
            gr.Dropdown(["Yes", "No"], label="Senior Citizen"),
            gr.Dropdown(["Yes", "No"], label="Partner"),
            gr.Dropdown(["Yes", "No"], label="Dependents"),
            gr.Dropdown(["Yes", "No"], label="Phone Service"),
            gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines"),
            gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service"),
            gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security"),
            gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup"),
            gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection"),
            gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support"),
            gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV"),
            gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies"),
            gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract"),
            gr.Dropdown(["Yes", "No"], label="Paperless Billing"),
            gr.Dropdown(
                ["Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"],
                label="Payment Method"
            ),
            gr.Number(label="Tenure (months)"),
            gr.Number(label="Monthly Charges"),
            gr.Number(label="Total Charges"),
        ],
        outputs="text",
        title="Telco Churn Predictor",
        description="Fill in the customer details to get a churn prediction.",
    )

    return gr.mount_gradio_app(app, demo, path)