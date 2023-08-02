from flask import Blueprint, render_template, redirect
import typing

views = Blueprint("views", __name__)

@views.route("/")
def Home() -> str:
	data = GetCommonData()

	return render_template("overview.html", **data)


@views.route("/metrics")
def Metrics() -> str:
	# redirect to wandb page
	return redirect("https://wandb.ai/louiej-s/Dissertation", code=302)


@views.route("/episodereview")
def EpisodeReview() -> str:
	data = GetCommonData()

	return render_template("episodereview.html", **data)


def GetCommonData() -> typing.Dict[str, typing.Any]:
	data = {}


	return data