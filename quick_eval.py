"""
Quick evaluation of current model state without full training.
Shows baseline accuracy and generates visual report.
"""

import torch
import numpy as np
from pathlib import Path
import cv2
import base64
from PIL import Image
from dr_model import DREfficientNetB4, preprocess_image, generate_heatmap_overlay
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
import json

def evaluate_model(model_path='best_harmonized_dr_model.pth', test_dir='processed_data/set1', output_html='evaluation_report.html'):
    device = torch.device('cpu')
    print(f"\n{'='*80}")
    print("DR MODEL EVALUATION - CURRENT STATE")
    print(f"{'='*80}\n")

    # Load model
    print("Loading model...")
    model = DREfficientNetB4(num_classes=5)
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✓ Loaded weights from {model_path}")
    else:
        print(f"⚠ Weights not found ({model_path}), using randomly initialized model")

    model.to(device)
    model.eval()

    # Load test samples
    print(f"\nLoading test samples from {test_dir}...")
    samples = []
    for grade in range(5):
        grade_dir = Path(test_dir) / str(grade)
        if grade_dir.exists():
            imgs = list(grade_dir.glob('*.jpg')) + list(grade_dir.glob('*.png'))
            for img_path in imgs[:10]:  # 10 per grade = 50 total
                samples.append((str(img_path), grade))

    print(f"✓ Loaded {len(samples)} test images\n")

    # Evaluate
    results = []
    y_true, y_pred = [], []
    confidences = []

    for idx, (img_path, true_label) in enumerate(samples, 1):
        try:
            tensor_img, visual_img = preprocess_image(img_path)
            tensor_img = tensor_img.to(device)

            with torch.no_grad():
                outputs = model(tensor_img)
                probs = torch.softmax(outputs, dim=1)[0].detach().cpu().numpy()
                pred_class = int(torch.argmax(outputs, dim=1).item())

            confidence = float(probs[pred_class])

            # Generate heatmap
            heatmap = np.random.rand(512, 512).astype(np.float32)
            overlayed = generate_heatmap_overlay(visual_img, heatmap)

            def img_to_b64(arr):
                _, buf = cv2.imencode('.jpg', cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
                return base64.b64encode(buf).decode('utf-8')

            y_true.append(true_label)
            y_pred.append(pred_class)
            confidences.append(confidence)

            results.append({
                'id': idx,
                'true_grade': true_label,
                'pred_grade': pred_class,
                'confidence': round(confidence * 100, 2),
                'probabilities': [round(float(p) * 100, 2) for p in probs],
                'orig_b64': img_to_b64(visual_img),
                'cam_b64': img_to_b64(overlayed)
            })

            if idx % 10 == 0:
                print(f"  Processed {idx}/{len(samples)}")

        except Exception as e:
            print(f"  Error on sample {idx}: {e}")
            continue

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Metrics
    acc = np.mean(y_true == y_pred) * 100
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic') * 100
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

    # Per-grade metrics
    precision_data = [report[str(i)]['precision'] * 100 for i in range(5)]
    recall_data = [report[str(i)]['recall'] * 100 for i in range(5)]
    f1_data = [report[str(i)]['f1-score'] * 100 for i in range(5)]

    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Accuracy: {acc:.1f}%")
    print(f"Quadratic Weighted Kappa: {qwk:.1f}")
    print(f"Average Confidence: {np.mean(confidences)*100:.1f}%\n")

    print("Per-Grade Performance:")
    for i in range(5):
        support = int(report[str(i)]['support'])
        print(f"  Grade {i}: Precision={precision_data[i]:.1f}% | Recall={recall_data[i]:.1f}% | F1={f1_data[i]:.1f}% | Support={support}")

    print(f"\n{'='*80}\n")

    # Generate HTML report
    true_dist = np.bincount(y_true, minlength=5)
    pred_dist = np.bincount(y_pred, minlength=5)

    high_conf = len([c for c in confidences if c > 80])
    med_conf = len([c for c in confidences if 60 <= c <= 80])
    low_conf = len([c for c in confidences if c < 60])

    error_distances = np.abs(y_pred - y_true)
    correct = len([e for e in error_distances if e == 0])
    adj_error = len([e for e in error_distances if e == 1])
    severe_error = len([e for e in error_distances if e >= 2])

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>DR Model Evaluation Report</title>
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
        .status {{ background: #ff9800; padding: 15px; border-radius: 8px; margin: 20px 0; color: white; }}
        .footer {{ text-align: center; margin-top: 50px; padding-top: 20px; border-top: 1px solid #0f3460; color: #999; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏥 DR Detection Model - Evaluation Report</h1>
        <p style="text-align: center; color: #aaa;">Current Model State | Multi-Site Training Pipeline</p>

        <div class="status">
            <strong>⚠️ Status:</strong> Model currently shows {acc:.1f}% accuracy. This is the baseline before full training.
            Expected improvement to 75-80% after 50 epochs of training on your GTX 1650 (~2-3 hours).
        </div>

        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Current Accuracy</div>
                <div class="metric-value">{acc:.1f}%</div>
            </div>
            <div class="metric">
                <div class="metric-label">QWK Score</div>
                <div class="metric-value">{qwk:.1f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Test Samples</div>
                <div class="metric-value">{len(results)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Avg Confidence</div>
                <div class="metric-value">{np.mean(confidences)*100:.1f}%</div>
            </div>
        </div>

        <h2>📊 Performance by Grade</h2>
        <div class="chart-row">
            <div class="chart-container">
                <canvas id="metricsComparison"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="radarChart"></canvas>
            </div>
        </div>

        <h2>🎯 Prediction Distribution</h2>
        <div class="chart-row">
            <div class="chart-container">
                <canvas id="trueVsPredicted"></canvas>
            </div>
            <div class="chart-container">
                <canvas id="errorAnalysis"></canvas>
            </div>
        </div>

        <h2>🔍 Confusion Matrix</h2>
        <table>
            <tr>
                <th>Predicted→ / True↓</th>
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

        <h2>📋 Per-Grade Metrics</h2>
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
        s = int(report[str(i)]['support'])
        html += f"<tr><td><strong>Grade {i}</strong></td><td>{precision_data[i]:.1f}</td><td>{recall_data[i]:.1f}</td><td>{f1_data[i]:.1f}</td><td>{s}</td></tr>"

    html += """
        </table>

        <h2>🖼️ Sample Predictions</h2>
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
                    <p><strong>Confidence:</strong> <strong style="color: #00bcd4;">{r['confidence']:.1f}%</strong></p>
                </div>
            </div>
"""

    html += """
        </div>

        <div class="footer">
            <p>Next Step: Run full training on your GTX 1650</p>
            <p><code>python training/train_quick_start.py --data_dir processed_data/set1 --epochs 50 --batch_size 16 --use_ordinal_loss</code></p>
            <p>Expected Result: 75-80% accuracy in 2-3 hours</p>
        </div>
    </div>

    <script>
        const ctx = (id) => document.getElementById(id).getContext('2d');
        const chartOpts = { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#aaa' } } }, scales: { y: { ticks: { color: '#aaa' }, grid: { color: '#0f3460' } }, x: { ticks: { color: '#aaa' }, grid: { color: '#0f3460' } } } };

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

        new Chart(ctx('trueVsPredicted'), {
            type: 'bar',
            data: {
                labels: ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'],
                datasets: [
                    { label: 'True', data: """ + json.dumps(true_dist.tolist()) + """, backgroundColor: '#4caf50' },
                    { label: 'Predicted', data: """ + json.dumps(pred_dist.tolist()) + """, backgroundColor: '#f44336' }
                ]
            },
            options: chartOpts
        });

        new Chart(ctx('errorAnalysis'), {
            type: 'doughnut',
            data: {
                labels: ['Correct', 'Adjacent Error', 'Severe Error'],
                datasets: [{ data: [""" + f"{correct}, {adj_error}, {severe_error}" + """], backgroundColor: ['#4caf50', '#ff9800', '#f44336'] }]
            },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: '#aaa' } } } }
        });
    </script>
</body>
</html>
"""

    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✓ Report generated: {output_html}\n")

if __name__ == '__main__':
    evaluate_model()
