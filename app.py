from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import numpy as np

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')

docs = [
    "Regular exercise improves cardiovascular health and boosts energy levels.",
    "Eating a balanced diet with fruits and vegetables is essential for good health.",
    "Drinking enough water daily keeps your body hydrated and improves skin health.",
    "Getting 7 to 8 hours of sleep every night helps your body recover and stay sharp.",
    "Meditation and mindfulness reduce stress and improve mental clarity.",
    "Python is a popular programming language used in data science and AI.",
    "Machine learning models learn patterns from large amounts of data.",
    "Vector databases store data as mathematical vectors for fast similarity search.",
    "Natural language processing helps computers understand and generate human text.",
    "Deep learning uses neural networks with many layers to solve complex problems.",
    "The Eiffel Tower is located in Paris, France and was built in 1889.",
    "The Amazon River is the largest river in the world by water discharge.",
    "Climate change is causing rising sea levels and more extreme weather events.",
    "Electric vehicles are becoming more popular as battery technology improves.",
    "Space exploration has led to many scientific discoveries and technological advances.",
]

embeddings = model.encode(docs, convert_to_numpy=True)

vectors = []
texts = []
for i in range(len(docs)):
    vectors.append(embeddings[i])
    texts.append(docs[i])

def search(query, k=3):
    q_vec = model.encode([query], convert_to_numpy=True)[0]
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-10)
    scores = []
    for i in range(len(vectors)):
        v = vectors[i] / (np.linalg.norm(vectors[i]) + 1e-10)
        sim = np.dot(q_norm, v)
        scores.append((float(sim), texts[i]))
    scores.sort(reverse=True)
    return scores[:k]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search_route():
    data = request.json
    query = data.get("query", "")
    if not query:
        return jsonify([])
    results = search(query)
    return jsonify([{"score": s, "text": t} for s, t in results])

if __name__ == "__main__":
    if __name__ == "__main__":
        app.run(debug=False)
