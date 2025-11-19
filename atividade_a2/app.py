from flask import Flask
from controller.routes import bp

app = Flask(__name__)

app.secret_key = '12345'

app.register_blueprint(bp)

if __name__ == '__main__':
    app.run(debug=True)
