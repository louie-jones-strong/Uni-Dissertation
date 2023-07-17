from flask import Blueprint, render_template

views = Blueprint("views", __name__)

@views.route("/")
def home() -> str:
	return render_template("index.html", name="Louie")