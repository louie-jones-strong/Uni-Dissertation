from flask import Blueprint, render_template, redirect

views = Blueprint("views", __name__)

@views.route("/")
def Home() -> str:
	return render_template("overview.html", name="Louie")


@views.route("/metrics")
def Metrics() -> str:
	# redirect to wandb page
	return redirect("https://wandb.ai/louiej-s/Dissertation", code=302)


@views.route("/episodereview")
def EpisodeReview() -> str:
	return render_template("episodereview.html", name="Louie")