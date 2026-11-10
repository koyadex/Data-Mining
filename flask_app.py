"""Web demo phân loại hoa Iris bằng cây quyết định đã huấn luyện."""

import math
import os

import joblib
from flask import Flask, abort, render_template, request, send_file


app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Nạp mô hình một lần khi ứng dụng khởi động.
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
labels = joblib.load(os.path.join(BASE_DIR, "labels.pkl"))

FIELDS = (
    ("sepal_length", "Chiều dài đài hoa"),
    ("sepal_width", "Chiều rộng đài hoa"),
    ("petal_length", "Chiều dài cánh hoa"),
    ("petal_width", "Chiều rộng cánh hoa"),
)
TREE_IMAGE = os.path.join(BASE_DIR, "cay_quyet_dinh.png")
SAMPLES = (
    {
        "name": "Iris-setosa",
        "values": (5.1, 3.5, 1.4, 0.2),
    },
    {
        "name": "Iris-versicolor",
        "values": (6.0, 2.9, 4.5, 1.5),
    },
    {
        "name": "Iris-virginica",
        "values": (6.7, 3.0, 5.8, 2.2),
    },
)


def explain_prediction(sample):
    """Trả về các điều kiện cây đã dùng để phân loại một mẫu."""
    tree = model.tree_
    rules = []

    for node in model.decision_path(sample).indices:
        if tree.children_left[node] == -1:
            break

        feature_index = tree.feature[node]
        field_name = FIELDS[feature_index][1]
        value = float(sample[0][feature_index])
        threshold = float(tree.threshold[node])
        operator = "≤" if value <= threshold else ">"
        rules.append(
            f"{field_name} ({value:g} cm) {operator} {threshold:.2f} cm"
        )

    return rules


@app.route("/tree-image", methods=["GET"])
def tree_image():
    """Phục vụ đúng ảnh cây quyết định trong thư mục dự án."""
    if not os.path.isfile(TREE_IMAGE):
        abort(404)
    return send_file(TREE_IMAGE, mimetype="image/png")


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    values = {field: "" for field, _ in FIELDS}

    if request.method == "POST":
        values = {
            field: request.form.get(field, "").strip()
            for field, _ in FIELDS
        }

        try:
            numbers = [float(values[field]) for field, _ in FIELDS]
            if any(not math.isfinite(number) or number <= 0 for number in numbers):
                raise ValueError
        except ValueError:
            error = "Vui lòng nhập đủ 4 số đo hữu hạn và lớn hơn 0 (đơn vị cm)."
        else:
            sample = [numbers]
            predicted_class = int(model.predict(sample)[0])
            probabilities = model.predict_proba(sample)[0]
            probability_rows = [
                {
                    "label": labels[int(class_id)],
                    "percent": round(float(probability) * 100, 1),
                }
                for class_id, probability in zip(model.classes_, probabilities)
            ]
            result = {
                "label": labels[predicted_class],
                "confidence": max(row["percent"] for row in probability_rows),
                "probabilities": probability_rows,
                "rules": explain_prediction(sample),
            }

    return render_template(
        "index.html",
        result=result,
        error=error,
        values=values,
        fields=FIELDS,
        samples=SAMPLES,
        has_tree_image=os.path.isfile(TREE_IMAGE),
    )


if __name__ == "__main__":
    app.run(debug=False)
