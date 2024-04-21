import math
from .f1_strategy_state import FOneStrategyState, ActionEnum
from .f1_strategy_observer import FOneStrategyObserver
import pyspiel

_MIN_PLAYERS = 3
_MAX_PLAYERS = 10
_NUM_PLAYERS = 3
_GAME_TYPE = pyspiel.GameType(
    short_name="python_f_one_strategy",
    long_name="Python F1 Strategy",
    dynamics=pyspiel.GameType.Dynamics.SIMULTANEOUS,
    chance_mode=pyspiel.GameType.ChanceMode.SAMPLED_STOCHASTIC,
    information=pyspiel.GameType.Information.PERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.GENERAL_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_MIN_PLAYERS,
    min_num_players=_MAX_PLAYERS,
    provides_information_state_string=False,
    provides_information_state_tensor=False,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=False)
_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=len(ActionEnum),
    max_chance_outcomes=math.factorial(_NUM_PLAYERS),  # Technically all transitions are possible
    num_players=_NUM_PLAYERS,
    min_utility=0.0,
    max_utility=float(_NUM_PLAYERS),
    utility_sum=(_NUM_PLAYERS-1)*(_NUM_PLAYERS-2)/2, # sum of integers up to n-1
    max_game_length=20)  # 20 laps

class FOneStrategyGame(pyspiel.Game):
    def __init__(self, params=None):
        super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())
        self.num_agents = _NUM_PLAYERS
        self.total_laps = _GAME_INFO.max_game_length

    def new_initial_state(self):
        return FOneStrategyState(self, num_agents=self.num_agents, total_laps=self.total_laps)
    
    def action_to_string(self, action: int):
        return ActionEnum(action).to_string()
    
    def deserialize_state(serialized_state):
        return FOneStrategyState.from_string(serialized_state)
    
    def max_chance_outcomes(self):
        return math.factorial(self.num_agents)
    
    def max_game_length(self):
        return self.total_laps
    
    def max_utility(self):
        return float(self.num_agents)
    
    def min_utility(self):
        return 0.0
    
    def num_distinct_actions(self):
        return len(ActionEnum)
    
    def observation_tensor_shape(self):
        return 3+5*self.num_agents  # TODO: Update this to select this dynamically
    
    def observation_tensor_size(self):
        return self.observation_tensor_shape()
    
    def make_py_observer(self, iig_obs_type=None, params=None):
        """Returns an object used for observing game state."""
        return FOneStrategyObserver(
            self,
            iig_obs_type or pyspiel.IIGObservationType(perfect_recall=True, public_info=True), 
            params
        )