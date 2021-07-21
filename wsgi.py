import webbrowser
from threading import Timer
from main import app

def open_browser():
    webbrowser.open_new('http://127.0.0.1:8080/')

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(host="127.0.0.1", port=8080, debug=True, use_reloader=False)
    count = 0