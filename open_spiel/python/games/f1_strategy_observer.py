import numpy as np

class FOneStrategyObserver:
  """Observer, conforming to the PyObserver interface."""
  def __init__(self, game, iig_obs_type, params=None) -> None:
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")
    self.game = game
    self.num_agents = game.num_agents
    # if not (iig_obs_type.public_info and iig_obs_type.perfect_recall):
    #   raise ValueError(f"Unsupported observation type {iig_obs_type}")

    total_size = 4 + 5 * game.num_agents
    self.tensor = np.zeros(total_size, dtype=np.float32)

    self.dict = {}
    """
    FormatSpec:
    The Tire Type is One Hot Encoded - Currently only one tire type is available
      - 0: Current Lap
      - 1: Total Laps
      - 2: Agent ID
      - 3-3+num_agents: Position of other agents by agent ID
      - 3+num_agents-3+2*num_agents: Progress of other agents by agent ID
      - 3+2*num_agents-3+3*num_agents: Tire Life of other agents by agent ID
      - 3+3*num_agents-3+4*num_agents: Tire Type of other agents by agent ID
      - 3+4*num_agents-3+5*num_agents: Last Action of other agents by agent ID
    """
    pieces = [
      ("current_lap", 1, (1,)),
      ("total_laps", 1, (1,)),
      ("agent_id", 1, (1,)),
      ("position", self.num_agents, (self.num_agents,)),
      ("progress", self.num_agents, (self.num_agents,)),
      ("tire_life", self.num_agents, (self.num_agents,)),
      ("tire_type", self.num_agents, (self.num_agents,)),
      ("last_action", self.num_agents, (self.num_agents,))
    ]
    index = 0
    for name, size, shape in pieces:
      self.dict[name] = self.tensor[index:index + size].reshape(shape)
      index += size


  def set_from(self, state, player):
    """Sets the observation tensor from the state for the specified player."""
    if player < 0:
      raise RuntimeError(f"player >= 0")
    elif player >= self.num_agents:
      raise RuntimeError("player <")
    self.tensor = state.observation_tensor(player)

  # def set_from(self, state):
  #   """Sets the observation tensor from the state."""
  #   self.tensor = state.observation_tensor()

  def string_from(self, state, player):
    """Returns a string representation of the observation for the specified player."""
    if player < 0:
      raise RuntimeError(f"player >= 0")
    elif player >= self.num_agents:
      raise RuntimeError("player <")
    return ''

  # def string_from(self, state):
  #   """Returns a string representation of the observation for the specified player."""
  #   return state.serialize()