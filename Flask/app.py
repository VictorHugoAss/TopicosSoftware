from flask import Flask, render_template

app = Flask(__name__)

# Página inicial
@app.route("/")
def index():
    return render_template("index.html")

# Página "Sobre"
@app.route("/sobre")
def about():
    return render_template("sobre.html")

if __name__ == "__main__":
    app.run(debug=True)
