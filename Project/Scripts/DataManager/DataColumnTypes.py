from typing import Any
import enum

class DataColumnTypes(enum.Enum):
	CurrentState = 0
	NextState = 1

	Action = 2

	Reward = 3
	MaxFutureRewards = 4

	Terminated = 5
	Truncated = 6

	PlayStyleTags = 7



def GetColumn(columnFilter:list[DataColumnTypes], rows:list[Any]) -> list[Any]:
	columns = []
	for col in columnFilter:

		assert col.value < len(rows), f"Column {col.name} not found in rows"
		columns.append(rows[col.value])
	return columns
