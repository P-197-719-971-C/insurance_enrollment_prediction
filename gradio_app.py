"""
Insurance Enrollment Predictor - Gradio UI
Clean, minimal implementation with full functionality.

Run: uv run python gradio_app.py
"""

import gradio as gr
import requests
import pandas as pd
from datetime import datetime
import tempfile
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# Configuration
API_URL = "http://localhost:8000"
MODELS = ["logistic_regression", "random_forest", "lightgbm"]
DEFAULT_MODEL = "logistic_regression"

MODEL_INFO = {
    "logistic_regression": "Best for interpretability and linear relationships",
    "random_forest": "Best for non-linear patterns and feature interactions",
    "lightgbm": "Best for complex patterns with L1/L2 regularization"
}

# Initialize history
history = pd.DataFrame(columns=[
    'Timestamp', 'Model', 'Has Dependents', 'Employment Type', 
    'Age', 'Salary', 'Prediction', 'Probability', 'Confidence'
])


def predict_single(has_dependents, employment_type, age, salary, model):
    """Make single prediction."""
    global history
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={
                "has_dependents": has_dependents,
                "employment_type": employment_type,
                "age": int(age),
                "salary": float(salary),
                "model": model
            },
            timeout=10
        )
        
        if response.ok:
            result = response.json()
            
            # Add to history
            new_row = pd.DataFrame([{
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Model': model.replace('_', ' ').title(),
                'Has Dependents': has_dependents,
                'Employment Type': employment_type,
                'Age': age,
                'Salary': salary,
                'Prediction': result['enrolled'],
                'Probability': f"{result['probability']:.4f}",
                'Confidence': result['confidence']
            }])
            history = pd.concat([history, new_row], ignore_index=True)
            
            # Format output
            status = "✅ WILL ENROLL" if result['prediction'] == 1 else "❌ WON'T ENROLL"
            
            return (
                f"## {status}\n\n"
                f"**Model:** {model.replace('_', ' ').title()}\n\n"
                f"**Probability:** {result['probability']:.4f} ({result['probability']*100:.2f}%)\n\n"
                f"**Confidence:** {result['confidence']}\n\n"
                f"---\n*{MODEL_INFO[model]}*",
                history,
                len(history)
            )
        else:
            return f"❌ API Error: {response.status_code}", history, len(history)
            
    except Exception as e:
        return f"❌ Error: {str(e)}", history, len(history)


def predict_batch(file, model):
    """Predict on uploaded file."""
    global history
    
    if file is None:
        return "❌ Please upload a file", history, len(history)
    
    try:
        # Read file
        df = pd.read_csv(file.name) if file.name.endswith('.csv') else pd.read_excel(file.name)
        
        # Validate columns
        required = ['has_dependents', 'employment_type', 'age', 'salary']
        missing = [col for col in required if col not in df.columns]
        if missing:
            return f"❌ Missing columns: {', '.join(missing)}", history, len(history)
        
        # Make request
        employees = df[required].to_dict('records')
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={"employees": employees, "model": model},
            timeout=60
        )
        
        if response.ok:
            result = response.json()
            predictions = result['predictions']
            
            # Add results to dataframe
            df['Prediction'] = [p['enrolled'] for p in predictions]
            df['Probability'] = [f"{p['probability']:.4f}" for p in predictions]
            df['Confidence'] = [p['confidence'] for p in predictions]
            df['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            df['Model'] = model.replace('_', ' ').title()
            
            # Rename and reorder columns
            df = df.rename(columns={
                'has_dependents': 'Has Dependents',
                'employment_type': 'Employment Type',
                'age': 'Age',
                'salary': 'Salary'
            })
            cols = ['Timestamp', 'Model', 'Has Dependents', 'Employment Type', 
                    'Age', 'Salary', 'Prediction', 'Probability', 'Confidence']
            df = df[cols]
            
            # Add to history
            history = pd.concat([history, df], ignore_index=True)
            
            summary = result['summary']
            return (
                f"## ✅ Batch Complete!\n\n"
                f"**Total:** {summary['total_employees']}\n\n"
                f"**Will Enroll:** {summary['predicted_enrolled']} ({summary['enrollment_rate']:.1f}%)\n\n"
                f"**Average Probability:** {summary['average_probability']:.4f}",
                history,
                len(history)
            )
        else:
            return f"❌ API Error: {response.status_code}", history, len(history)
            
    except Exception as e:
        return f"❌ Error: {str(e)}", history, len(history)


def save_template():
    """Generate sample template."""
    sample = pd.DataFrame([
        {'has_dependents': 'Yes', 'employment_type': 'Full-time', 'age': 35, 'salary': 75000},
        {'has_dependents': 'No', 'employment_type': 'Part-time', 'age': 25, 'salary': 35000},
        {'has_dependents': 'Yes', 'employment_type': 'Contract', 'age': 45, 'salary': 95000},
    ])
    
    fd, path = tempfile.mkstemp(suffix='.xlsx', prefix='template_')
    os.close(fd)
    sample.to_excel(path, index=False, engine='openpyxl')
    return gr.File(value=path, visible=True)


def save_excel():
    """Export history as Excel."""
    if len(history) == 0:
        return gr.File(visible=False)
    
    fd, path = tempfile.mkstemp(suffix='.xlsx', prefix='predictions_')
    os.close(fd)
    history.to_excel(path, index=False, engine='openpyxl')
    return gr.File(value=path, visible=True)


def save_csv():
    """Export history as CSV."""
    if len(history) == 0:
        return gr.File(visible=False)
    
    fd, path = tempfile.mkstemp(suffix='.csv', prefix='predictions_')
    os.close(fd)
    history.to_csv(path, index=False)
    return gr.File(value=path, visible=True)


def save_pdf():
    """Export history as PDF."""
    if len(history) == 0:
        return gr.File(visible=False)
    
    fd, path = tempfile.mkstemp(suffix='.pdf', prefix='predictions_')
    os.close(fd)
    
    doc = SimpleDocTemplate(path, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    elements.append(Paragraph("Insurance Enrollment Predictions", styles['Title']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Summary
    summary = f"Total: {len(history)} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    elements.append(Paragraph(summary, styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))
    
    # Table
    data = [history.columns.tolist()] + history.values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 7),
    ]))
    
    elements.append(table)
    doc.build(elements)
    return gr.File(value=path, visible=True)


def clear_history():
    """Clear all history."""
    global history
    history = pd.DataFrame(columns=[
        'Timestamp', 'Model', 'Has Dependents', 'Employment Type', 
        'Age', 'Salary', 'Prediction', 'Probability', 'Confidence'
    ])
    return history, 0, "✅ History cleared"


# Build UI
with gr.Blocks(title="Insurance Enrollment Predictor", theme=gr.themes.Soft()) as app:
    
    gr.Markdown(
        """
        # 🏥 Insurance Enrollment Predictor
        ### Compare predictions across 3 ML models
        """
    )
    
    with gr.Tabs():
        
        # Single Prediction
        with gr.Tab("🔮 Single Prediction"):
            with gr.Row():
                with gr.Column():
                    model_single = gr.Dropdown(
                        choices=MODELS,
                        value=DEFAULT_MODEL,
                        label="Select Model"
                    )
                    model_desc = gr.Markdown(MODEL_INFO[DEFAULT_MODEL])
                    
                    gr.Markdown("---")
                    
                    has_dep = gr.Radio(["Yes", "No"], label="Has Dependents", value="Yes")
                    emp_type = gr.Dropdown(
                        ["Full-time", "Part-time", "Contract"], 
                        label="Employment Type", 
                        value="Full-time"
                    )
                    age = gr.Slider(18, 100, value=35, label="Age")
                    salary = gr.Number(label="Annual Salary ($)", value=75000)
                    
                    predict_btn = gr.Button("🔮 Predict", variant="primary", size="lg")
                
                with gr.Column():
                    result = gr.Markdown()
                    
                    gr.Examples(
                        examples=[
                            ["Yes", "Full-time", 35, 75000],
                            ["No", "Part-time", 25, 35000],
                            ["Yes", "Contract", 45, 95000],
                        ],
                        inputs=[has_dep, emp_type, age, salary],
                        label="Quick Examples"
                    )
        
        # Batch Prediction
        with gr.Tab("📊 Batch Prediction"):
            model_batch = gr.Dropdown(
                choices=MODELS,
                value=DEFAULT_MODEL,
                label="Select Model"
            )
            batch_desc = gr.Markdown(MODEL_INFO[DEFAULT_MODEL])
            
            gr.Markdown(
                """
                **Required Columns:** `has_dependents`, `employment_type`, `age`, `salary`
                """
            )
            
            template_btn = gr.Button("📥 Download Template")
            template_file = gr.File(label="Template", visible=False)
            
            file_upload = gr.File(label="Upload CSV or Excel", file_types=['.csv', '.xlsx', '.xls'])
            batch_btn = gr.Button("🚀 Process Batch", variant="primary", size="lg")
            batch_result = gr.Markdown()
        
        # History
        with gr.Tab("📈 History"):
            with gr.Row():
                record_count = gr.Number(label="Total Records", value=0, interactive=False)
                clear_btn = gr.Button("🗑️ Clear History", variant="stop")
            
            clear_msg = gr.Markdown()
            history_table = gr.Dataframe(value=history, label="Prediction History")
            
            gr.Markdown("### Download")
            with gr.Row():
                excel_btn = gr.Button("📗 Excel", variant="primary")
                csv_btn = gr.Button("📄 CSV")
                pdf_btn = gr.Button("📕 PDF")
            
            excel_file = gr.File(visible=False)
            csv_file = gr.File(visible=False)
            pdf_file = gr.File(visible=False)
    
    gr.Markdown(
        """
        ---
        💡 **Tip:** Try the same employee data with different models to compare predictions!
        
        🔗 [API Documentation](http://localhost:8000/docs)
        """
    )
    
    # Events
    model_single.change(
        fn=lambda m: MODEL_INFO[m],
        inputs=model_single,
        outputs=model_desc
    )
    
    model_batch.change(
        fn=lambda m: MODEL_INFO[m],
        inputs=model_batch,
        outputs=batch_desc
    )
    
    predict_btn.click(
        fn=predict_single,
        inputs=[has_dep, emp_type, age, salary, model_single],
        outputs=[result, history_table, record_count]
    )
    
    batch_btn.click(
        fn=predict_batch,
        inputs=[file_upload, model_batch],
        outputs=[batch_result, history_table, record_count]
    )
    
    clear_btn.click(
        fn=clear_history,
        outputs=[history_table, record_count, clear_msg]
    )
    
    template_btn.click(fn=save_template, outputs=template_file)
    excel_btn.click(fn=save_excel, outputs=excel_file)
    csv_btn.click(fn=save_csv, outputs=csv_file)
    pdf_btn.click(fn=save_pdf, outputs=pdf_file)


if __name__ == "__main__":
    print("🚀 Starting Insurance Enrollment Predictor")
    print(f"🔗 API: {API_URL}")
    print(f"🤖 Models: {', '.join(MODELS)}")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )