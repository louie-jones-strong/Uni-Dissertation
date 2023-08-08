import src.Common.Store.ExperienceStore.EsBase as EsBase
import reverb
import src.Common.Enums.eDataColumnTypes as DCT

CurrentState_Name = DCT.eDataColumnTypes.CurrentState.name
NextState_Name = DCT.eDataColumnTypes.NextState.name
Action_Name = DCT.eDataColumnTypes.Action.name
Reward_Name = DCT.eDataColumnTypes.Reward.name
MaxFutureRewards_Name = DCT.eDataColumnTypes.MaxFutureRewards.name
Terminated_Name = DCT.eDataColumnTypes.Terminated.name
Truncated_Name = DCT.eDataColumnTypes.Truncated.name

class EsReverb(EsBase.EsBase):

	def __init__(self) -> None:
		super().__init__()

		self.TableTrajectoriesLens = {
			"Forward_Trajectories": 1,
			"Value_Trajectories": 1,
			"Human_Trajectories": 1,
		}
		return


# region Reverb Client

	_ReverbConnection = None

	def Get_ReverbConnection(self):
		if type(self)._ReverbConnection is None:
			type(self)._ReverbConnection = reverb.Client(f"experience-store:{5001}")

		return type(self)._ReverbConnection

	# def Set_ReverbConnection(self, val):
	# 	type(self)._ReverbConnection = val
	# 	return

	ReverbConnection = property(Get_ReverbConnection)

# endregion Reverb Client


	def EmptyTransitionBuffer(self) -> None:

		numTransitions = len(self._TransitionBuffer)
		with self.ReverbConnection.trajectory_writer(num_keep_alive_refs=numTransitions) as writer:

			for i in range(numTransitions):

				transition = self.PopTransition()
				state, nextState, action, reward, futureRewards, terminated, truncated = transition


				writer.append({
					CurrentState_Name: state,
					NextState_Name: nextState,
					Action_Name: action,
					Reward_Name: reward,
					MaxFutureRewards_Name: futureRewards,
					Terminated_Name: terminated,
					Truncated_Name: truncated
				})

				for tableKey, tableLen in self.TableTrajectoriesLens.items():
					if i+1 >= tableLen:
						writer.create_item(
							table=tableKey,
							priority=1,
							trajectory={
								CurrentState_Name: writer.history[CurrentState_Name][-tableLen:],
								NextState_Name: writer.history[NextState_Name][-tableLen:],
								Action_Name: writer.history[Action_Name][-tableLen:],
								Reward_Name: writer.history[Reward_Name][-tableLen:],
								MaxFutureRewards_Name: writer.history[MaxFutureRewards_Name][-tableLen:],
								Terminated_Name: writer.history[Terminated_Name][-tableLen:],
								Truncated_Name: writer.history[Truncated_Name][-tableLen:],
							})


			writer.end_episode(timeout_ms=1000)


		assert len(self._TransitionBuffer) == 0

		super().EmptyTransitionBuffer()
		return

	def UpdatePriorities(self, dataTable, keys, priorities) -> None:
		trajectoryPriorities = {}
		for i in range(len(keys)):

			key = int(keys[i])
			priority = priorities[i]

			trajectoryPriorities[key] = priority


		self.ReverbConnection.mutate_priorities(dataTable, trajectoryPriorities)
		return