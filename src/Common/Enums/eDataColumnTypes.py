from typing import Any
import typing
import enum

class eDataColumnTypes(enum.Enum):
	CurrentState = 0
	NextState = 1

	Action = 2

	Reward = 3
	MaxFutureRewards = 4

	Terminated = 5
	Truncated = 6

	PlayStyleTags = 7



def FilterColumns(columnFilter:typing.List[eDataColumnTypes], rows:typing.Tuple[Any, ...]) -> typing.List[Any]:
	columns = []
	for col in columnFilter:

		assert col.value < len(rows), f"Column {col.name} not found in rows"
		columns.append(rows[col.value])
	return columns


def FilterDict(columnFilter:typing.List[eDataColumnTypes], rowsDict) -> typing.List[Any]:
	columns = []
	for col in columnFilter:

		columns.append(rowsDict[col.name])
	return columns

