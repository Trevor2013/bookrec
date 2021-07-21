import os
import webbrowser
from threading import Timer
from flask_login import logout_user, current_user, login_required, login_manager, LoginManager, login_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from Trainer import get_author, getuserID, get_random_title, add_rating, prediction, books
from flask import Flask, render_template, redirect, url_for, request, session, flash, make_response
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine
import pandas as pd


# Function to open browser
def open_browser():
    webbrowser.open_new('http://127.0.0.1:8080/')


# Define app parameters and folder locations
image_folder = os.path.join('static', 'images')
app = Flask(__name__, template_folder="templates")
app.config['image_folder'] = image_folder

# Set up login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize user database
db = SQLAlchemy()
db.init_app(app)
engine = create_engine('sqlite:///users.db', echo=True)
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY


# Initialize variables
new_rating = pd.DataFrame(columns=['user_id', 'book_id', 'rating'])
newId = getuserID()
title1 = ""
title2 = ""
title3 = ""
title4 = ""
title5 = ""
title6 = ""
title7 = ""
title8 = ""
title9 = ""
title10 = ""
author1 = ""
author2 = ""
author3 = ""
author4 = ""
author5 = ""
author6 = ""
author7 = ""
author8 = ""
author9 = ""
author10 = ""
img1 = ""
img2 = ""
img3 = ""
img4 = ""
img5 = ""
img6 = ""
img7 = ""
img8 = ""
img9 = ""
img10 = ""


# LoginForm class
class LoginForm(FlaskForm):
    username = StringField('Username: ')
    password = PasswordField('Password: ')
    submit = SubmitField('Submit')


# User Class
class User(db.Model):
    __tablename__ = 'user'

    username = db.Column(db.String, primary_key=True)
    password = db.Column(db.String)
    authenticated = db.Column(db.Boolean, default=False)

    def is_active(self):
        return True

    def get_id(self):
        return self.username

    def is_authenticated(self):
        return self.authenticated

    def is_anonymous(self):
        return False


# User loader function
def user_loader(self, callback):
    self._user_callback = callback
    return callback


@login_manager.user_loader
def user_loader(user_id):
    return User.query.get(user_id)


# Create 'admin' user
with app.app_context():
    db.create_all()
    user = User(username='admin', password='admin', authenticated=False)
    user.authenticated = False
    db.session.add(user)
    db.session.commit()
    users = User.query.all()
    for row in users:
        print(row)


# Login screen
@app.route("/", methods=["GET", "POST"])
def login():
    error = None
    form = LoginForm()
    if form.validate_on_submit():
        user1 = User.query.get(form.username.data)
        if user1:
            if user1.password == form.password.data:
                user1.authenticated = True
                db.session.add(user1)
                db.session.commit()
                login_user(user1, remember=False)
                return redirect('/landing')
        else:
            error = 'Incorrect credentials.  Please try again.'
    return render_template("Welcome.html", error=error, form=form)

# landing (main menu) screen
@app.route("/landing")
@login_required
def menu():
    return render_template("Landing.html")


# Book rating screen
@app.route("/ratebooks", methods=["GET", "POST"])
@login_required
def second():
    msg = None
    global new_rating
    random_title, random_id, imgurl = get_random_title(new_rating['book_id'].to_numpy())
    author = get_author(random_id)
    if request.method == "POST":
        if request.form['submit'] == 'submit':
            rating = request.form.get('rating')
            if rating == "NA":
                return redirect("/ratebooks")
            else:
                random_title, random_id, imgurl = get_random_title(new_rating['book_id'].to_numpy())
                author = get_author(random_id)
                new_rating = new_rating.append({"user_id": newId, "book_id": random_id, "rating": rating},
                                               ignore_index=True)
                msg = "Rating accepted!  You have submitted " + str(len(new_rating)) + " rating(s)."
            return render_template("RateBooks.html", msg=msg, title=random_title, author=author, imgurl=imgurl)
        elif request.form['submit'] == 'getrecs':
            return redirect('loading')
    return render_template("RateBooks.html", title=random_title, author=author, imgurl=imgurl)



@app.route("/loading", methods=["GET", "POST"])
@login_required
def loading():
    return render_template("Loading.html")



# Book recommendation screen
@app.route("/recommendations", methods=["GET", "POST"])
@login_required
def third():
    # if request.method == 'GET':
    global title1, title2, title3, title4, title5, title6, title7, title8, title9, title10
    global img1, img2, img3, img4, img5, img6, img7, img8, img9, img10
    global author1, author2, author3, author4, author5, author6, author7, author8, author9, author10
    title1, title2, title3, title4, title5, title6, title7, title8, title9, title10, author1, author2, author3, \
    author4, author5, author6, author7, author8, author9, author10, img1, img2, img3, img4, img5, img6, img7, \
    img8, img9, img10 = prediction(newId, new_rating)
    return render_template("Recommendations.html", title1=title1, title2=title2, title3=title3, title4=title4,
                           title5=title5, title6=title6, title7=title7, title8=title8, title9=title9, title10=title10,
                           author1=author1, author2=author2, author3=author3, author4=author4, author5=author5,
                           author6=author6, author7=author7, author8=author8, author9=author9, author10=author10,
                           img1=img1, img2=img2, img3=img3, img4=img4, img5=img5, img6=img6, img7=img7,
                           img8=img8, img9=img9, img10=img10)
    # return make_response('POST request successful', 200)


# Data visualization dashboard screen
@app.route('/viewdata')
@login_required
def data():
    plt1_filename = os.path.join(app.config['image_folder'], 'plt1.png')
    plt2_filename = os.path.join(app.config['image_folder'], 'plt2.png')
    plt3_filename = os.path.join(app.config['image_folder'], 'plt3.png')
    return render_template("Data.html", plt1=plt1_filename, plt2=plt2_filename, plt3=plt3_filename)


# Shutdown app.  This deauthenticates the user, reinitializes certain variables, and returns the user to the login page
@app.route('/shutdown', methods=['GET'])
@login_required
def shutdown():
    user = current_user
    user.authenticated = False
    db.session.add(user)
    db.session.commit()
    logout_user()
    global new_rating
    del new_rating
    new_rating = pd.DataFrame(columns=['user_id', 'book_id', 'rating'])
    return redirect("/")


# Start up application
if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(host="127.0.0.1", port=8080, debug=True, use_reloader=False)
    count = 0
