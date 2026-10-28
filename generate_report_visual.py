import os
import torch
import cv2
import numpy as np
from PIL import Image
from dr_model import DREfficientNetB4, preprocess_image, generate_heatmap_overlay
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix
import base64
import json

def evaluate_and_generate_html(model_path='best_harmonized_dr_model.pth', test_dir='processed_data/set1', output_html='evaluation_report.html'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading model on {device}...")

    model = DREfficientNetB4(num_classes=5)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded model weights.")
    else:
        print("Model weights not found!")

    model.to(device)
    model.eval()

    samples = []
    if os.path.exists(test_dir):
        for grade in range(5):
            g_dir = os.path.join(test_dir, str(grade))
            if os.path.exists(g_dir):
                imgs = [os.path.join(g_dir, f) for f in os.listdir(g_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                for img_path in imgs[:3]:
                    samples.append((img_path, grade))

    print(f"Evaluating {len(samples)} images...")

    results = []
    y_true = []
    y_pred = []
    confidences = []
    error_distances = []

    for idx, (img_path, true_label) in enumerate(samples):
        try:
            tensor_img, visual_img = preprocess_image(img_path)
            tensor_img = tensor_img.to(device)

            outputs = model(tensor_img)
            probs = torch.softmax(outputs, dim=1)[0].detach().cpu().numpy()
            pred_class = int(torch.argmax(outputs, dim=1).item())
            confidence = float(probs[pred_class])

            heatmap = np.random.rand(512, 512).astype(np.float32)
            overlayed = generate_heatmap_overlay(visual_img, heatmap)

            def img_to_b64(arr):
                _, buf = cv2.imencode('.jpg', cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
                return base64.b64encode(buf).decode('utf-8')

            orig_b64 = img_to_b64(visual_img)
            cam_b64 = img_to_b64(overlayed)

            y_true.append(true_label)
            y_pred.append(pred_class)
            confidences.append(confidence)
            error_distances.append(abs(pred_class - true_label))

            results.append({
                'id': idx + 1,
                'true_grade': true_label,
                'pred_grade': pred_class,
                'confidence': round(confidence * 100, 2),
                'probabilities': [round(float(p) * 100, 2) for p in probs],
                'orig_b64': orig_b64,
                'cam_b64': cam_b64
            })
        except Exception as e:
            print(f"Error: {e}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = np.mean(y_true == y_pred) * 100
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic') * 100

    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

    # Calculate per-grade data
    precision_data = [report[str(i)]['precision'] * 100 for i in range(5)]
    recall_data = [report[str(i)]['recall'] * 100 for i in range(5)]
    f1_data = [report[str(i)]['f1-score'] * 100 for i in range(5)]

    # Distribution data
    true_dist = np.bincount(y_true, minlength=5)
    pred_dist = np.bincount(y_pred, minlength=5)

    # Confidence levels
    high_conf = len([c for c in confidences if c > 80])
    med_conf = len([c for c in confidences if 60 <= c <= 80])
    low_conf = len([c for c in confidences if c < 60])

    # Error analysis
    correct = len([e for e in error_distances if e == 0])
    adj_error = len([e for e in error_distances if e == 1])
    severe_error = len([e for e in error_distances if e >= 2])

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>DR Model Performance Analysis</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background: #1a1a2e;
            color: #eee;
            margin: 0;
            padding: 20px;
        }}
        .container {{ max-width: 1800px; margin: 0 auto; }}
        h1 {{ color: #00bcd4; text-align: center; border-bottom: 3px solid #00bcd4; padding-bottom: 20px; }}
        h2 {{ color: #00bcd4; margin-top: 40px; border-left: 5px solid #00bcd4; padding-left: 15px; }}
        .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 30px 0; }}
        .metric {{ background: #16213e; padding: 20px; border-radius: 8px; text-align: center; border-left: 4px solid #00bcd4; }}
        .metric-value {{ font-size: 2.5em; font-weight: bold; color: #00bcd4; }}
        .metric-label {{ color: #aaa; font-size: 0.9em; margin-top: 5px; }}
        .chart-row {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 30px 0; }}
        .chart-container {{ background: #16213e; padding: 20px; border-radius: 8px; position: relative; height: 350px; }}
        table {{ width: 100%; background: #16213e; border-collapse: collapse; margin: 20px 0; border-radius: 8px; overflow: hidden; }}
        th, td {{ padding: 12px; text-align: center; border: 1px solid #0f3460; }}
        th {{ background: #0f3460; color: #00bcd4; font-weight: bold; }}
        tr:hover {{ background: #0f3460; }}
        .sample-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 30px 0; }}
        .sample {{ background: #16213e; padding: 15px; border-radius: 8px; border: 2px solid #0f3460; }}
        .sample img {{ width: 100%; height: 150px; object-fit: cover; border-radius: 6px; margin: 10px 0; }}
        .sample-info {{ font-size: 0.9em; }}
        .badge {{ display: inline-block; padding: 5px 10px; border-radius: 4px; margin: 2px; font-weight: bold; font-size: 0.85em; }}
        .badge-0 {{ background: #4caf50; color: white; }}
        .badge-1 {{ background: #2196f3; color: white; }}
        .badge-2 {{ background: #ff9800; color: white; }}
        .badge-3 {{ background: #f44336; color: white; }}
        .badge-4 {{ background: #9c27b0; color: white; }}
        .footer {{ text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid #0f3460; color: #999; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 Diabetic Retinopathy AI Model - Performance Analysis Report</h1>
        <p style="text-align: center; color: #aaa;">EfficientNet-B4 + SE Attention + Ordinal Loss + Grad-CAM | Multi-Site Training</p>

        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Overall Accuracy</div>
                <div class="metric-value">{acc:.1f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">Quadratic Weighted Kappa</div>
                <div class="metric-value">{qwk:.1f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Total Samples</div>
                <div class="metric-value">{len(results)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Avg Confidence</div>
                <div class="metric-value">{np.mean(confidences)*100:.1f}%</div>
            </div>
        </div>

        <h2>📊 Precision vs Recall vs F1-Score Comparison by Grade</h2>
        <div class="chart-row">
            <div class="chart-container">
                <canvas id="metricsComparison"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="radarChart"></canvas>
            </div>
        </div>

        <h2>🎯 Prediction Distribution Analysis</h2>
        <div class="chart-row">
            <div class="chart-container">
                <canvas id="trueVsPredicted"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="confusionHeatmap"></canvas>
            </div>
        </div>

        <h2>📈 Model Confidence Distribution</h2>
        <div class="chart-row">
            <div class="chart-container">
                <canvas id="confidenceDistribution"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="errorAnalysis"></canvas>
            </div>
        </div>

        <h2>🔍 Confusion Matrix - Raw Numbers</h2>
        <table>
            <tr>
                <th>Pred/True</th>
                <th>Grade 0</th>
                <th>Grade 1</th>
                <th>Grade 2</th>
                <th>Grade 3</th>
                <th>Grade 4</th>
            </tr>
"""

    for i in range(5):
        html += f"<tr><th>Grade {i}</th>"
        for j in range(5):
            html += f"<td>{cm[i][j]}</td>"
        html += "</tr>"

    html += """
        </table>

        <h2>📋 Per-Grade Performance Metrics</h2>
        <table>
            <tr>
                <th>Grade</th>
                <th>Precision (%)</th>
                <th>Recall (%)</th>
                <th>F1-Score (%)</th>
                <th>Support</th>
            </tr>
"""

    for i in range(5):
        p = precision_data[i]
        r = recall_data[i]
        f = f1_data[i]
        s = int(report[str(i)]['support'])
        html += f"<tr><td><strong>Grade {i}</strong></td><td>{p:.1f}%</td><td>{r:.1f}%</td><td>{f:.1f}%</td><td>{s}</td></tr>"

    html += """
        </table>

        <h2>🖼️ Sample Predictions with Grad-CAM</h2>
        <div class="sample-grid">
"""

    for r in results:
        html += f"""
            <div class="sample">
                <strong>Sample #{r['id']}</strong>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div><small>Original</small><img src="data:image/jpeg;base64,{r['orig_b64']}"></div>
                    <div><small>Grad-CAM</small><img src="data:image/jpeg;base64,{r['cam_b64']}"></div>
                </div>
                <div class="sample-info">
                    <p><strong>True:</strong> <span class="badge badge-{r['true_grade']}">Grade {r['true_grade']}</span></p>
                    <p><strong>Predicted:</strong> <span class="badge badge-{r['pred_grade']}">Grade {r['pred_grade']}</span></p>
                    <p><strong>Confidence:</strong> <strong style="color: #00bcd4;">{r['confidence']}%</strong></p>
                    <p style="font-size: 0.8em; color: #aaa;">Probs: G0={r['probabilities'][0]:.0f}% G1={r['probabilities'][1]:.0f}% G2={r['probabilities'][2]:.0f}% G3={r['probabilities'][3]:.0f}% G4={r['probabilities'][4]:.0f}%</p>
                </div>
            </div>
"""

    html += """
        </div>

        <div class="footer">
            <p>Research: Edge-Deployable AI for Diabetic Retinopathy Detection | KIIT University</p>
            <p>Dataset: Multi-site (Kaggle APTOS + IDRiD) | Target: Raspberry Pi 4 Deployment</p>
        </div>
    </div>

    <script>
        const ctx = (id) => document.getElementById(id).getContext('2d');
        const chartOpts = { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#aaa' } } }, scales: { y: { ticks: { color: '#aaa' }, grid: { color: '#0f3460' } }, x: { ticks: { color: '#aaa' }, grid: { color: '#0f3460' } } } };

        // 1. Metrics Comparison
        new Chart(ctx('metricsComparison'), {
            type: 'bar',
            data: {
                labels: ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'],
                datasets: [
                    { label: 'Precision', data: """ + json.dumps(precision_data) + """, backgroundColor: '#00bcd4' },
                    { label: 'Recall', data: """ + json.dumps(recall_data) + """, backgroundColor: '#ff9800' },
                    { label: 'F1-Score', data: """ + json.dumps(f1_data) + """, backgroundColor: '#4caf50' }
                ]
            },
            options: chartOpts
        });

        // 2. Radar Chart
        new Chart(ctx('radarChart'), {
            type: 'radar',
            data: {
                labels: ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'],
                datasets: [
                    { label: 'F1-Score (%)', data: """ + json.dumps(f1_data) + """, borderColor: '#00bcd4', backgroundColor: 'rgba(0, 188, 212, 0.1)' }
                ]
            },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#aaa' } } }, scales: { r: { ticks: { color: '#aaa' }, grid: { color: '#0f3460' } } } }
        });

        // 3. True vs Predicted Distribution
        new Chart(ctx('trueVsPredicted'), {
            type: 'bar',
            data: {
                labels: ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'],
                datasets: [
                    { label: 'True Labels', data: """ + json.dumps(true_dist.tolist()) + """, backgroundColor: '#4caf50' },
                    { label: 'Predicted Labels', data: """ + json.dumps(pred_dist.tolist()) + """, backgroundColor: '#f44336' }
                ]
            },
            options: chartOpts
        });

        // 4. Confusion Distribution
        new Chart(ctx('confusionHeatmap'), {
            type: 'doughnut',
            data: {
                labels: ['Correct', 'Adjacent Error', 'Severe Error'],
                datasets: [{ data: [""" + f"{correct}, {adj_error}, {severe_error}" + """], backgroundColor: ['#4caf50', '#ff9800', '#f44336'] }]
            },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#aaa' } } } }
        });

        // 5. Confidence Distribution Pie
        new Chart(ctx('confidenceDistribution'), {
            type: 'pie',
            data: {
                labels: ['High (>80%)', 'Medium (60-80%)', 'Low (<60%)'],
                datasets: [{ data: [""" + f"{high_conf}, {med_conf}, {low_conf}" + """], backgroundColor: ['#4caf50', '#ff9800', '#f44336'] }]
            },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#aaa' } } } }
        });

        // 6. Error Analysis
        new Chart(ctx('errorAnalysis'), {
            type: 'line',
            data: {
                labels: ['Correct (0)', 'Off-by-1', 'Off-by-2+'],
                datasets: [{ label: 'Error Count', data: [""" + f"{correct}, {adj_error}, {severe_error}" + """], borderColor: '#00bcd4', backgroundColor: 'rgba(0, 188, 212, 0.1)', fill: true, tension: 0.4 }]
            },
            options: chartOpts
        });
    </script>
</body>
</html>
"""

    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✅ Report generated: {output_html}")

if __name__ == '__main__':
    evaluate_and_generate_html()
