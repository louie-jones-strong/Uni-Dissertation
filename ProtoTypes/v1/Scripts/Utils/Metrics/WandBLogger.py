from . import BaseLogger
import wandb


class WandBMetrics(BaseLogger.BaseLogger):
	_ProjectName = "Dissertation"

	def __init__(self, runId=None):
		super().__init__(runId=runId)

		wandb.init(project=self._ProjectName, id=runId, resume="allow")
		return

