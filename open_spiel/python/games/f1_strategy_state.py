from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np
import pyspiel

from .f1_strategy_params import *


@dataclass
class AgentData:
    agent_id: int
    tire_type: TireType
    tire_life: np.float32
    name: str
    attack_factor: np.float32
    defense_factor: np.float32
    position: int
    progress: np.float32
    terminated: bool = False
    truncated: bool = False
    current_move: ActionEnum = None

    @property
    def active(self):
        return not self.terminated and not self.truncated
    
    def todict(self):
        return {
            "agent_id": self.agent_id,
            "tire_type": self.tire_type.name,
            "tire_life": self.tire_life,
            "name": self.name,
            "attack_factor": self.attack_factor,
            "defense_factor": self.defense_factor,
            "position": self.position,
            "progress": self.progress,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "current_move": self.current_move.name if self.current_move is not None else None
        }


class AgentSet:
    def __init__(self, num_agents: int, agent_names: Union[List[str], None]) -> None:
        self.num_agents = num_agents
        self.agent_names = agent_names
        if self.agent_names is None:
            self.agent_names = [f"Agent_{i}" for i in range(self.num_agents)]
        self.agent_data = dict()
        self.create_agents()

    def create_agents(self):
        self.agent_data = {
            agent_idx: AgentData(
                agent_id=agent_idx,
                tire_type=TireType.MEDIUM,
                tire_life=MAX_TIRE_LIFE,
                name=self.agent_names[agent_idx],
                attack_factor=BASE_ATTACK_FACTOR,
                defense_factor=BASE_DEFENSE_FACTOR,
                position=-1,
                progress=0,
                terminated=False,
                truncated=False,
            )
            for agent_idx in range(self.num_agents)
        }

    def __getitem__(self, agent_idx: int):
        return self.agent_data[agent_idx]

    def __setitem__(self, agent_idx: int, agent_data: AgentData):
        self.agent_data[agent_idx] = agent_data

    def __contains__(self, agent_idx: int):
        return agent_idx in [i for i in self.agent_data.keys()]

    def __copy__(self):
        new_state = AgentSet(self.num_agents, self.agent_names)
        new_state.agent_data = deepcopy(self.agent_data)
        return new_state
    
    def __len__(self):
        return self.num_agents

    def get_sliced_copy(self, agent_indices: List[int]):
        new_state = AgentSet(
            len(agent_indices), [self.agent_names[i] for i in agent_indices]
        )
        new_state.agent_data = {i: self.agent_data[i] for i in agent_indices}
        # We need to update the positions of the agents in the new state
        index_by_position = new_state.agents_by_position.keys()
        for new_pos, index in enumerate(index_by_position):
            new_state.agent_data[list(new_state.agent_data.keys())[index]].position = (
                new_pos
            )
        return new_state

    def get_agent_in_position(self, position):
        return self.agent_data[self.agents_by_position[position]]

    @property
    def positions_by_agent(self):
        """The current positions of the agents. This is a dict of positions, keyed by agent index."""
        # Assemble positions from the agent data
        return {agent: self.agent_data[agent].position for agent in self.agent_data}

    @positions_by_agent.setter
    def positions_by_agent(self, new_positions: Dict[int, int]):
        """Set the positions of the agents. This will update the agent data."""
        for agent in new_positions:
            self.agent_data[agent].position = new_positions[agent]

    @property
    def agents_by_position(self):
        """The current positions of the agents. This is a dict of agent indices, keyed by position."""
        # Assemble positions from the agent data
        agents_in_order = np.argsort(
            [self.agent_data[agent].position for agent in self.agent_data]
        )
        return {
            position: list(self.agent_data.keys())[agent_key_index]
            for position, agent_key_index in enumerate(agents_in_order)
        }

    @agents_by_position.setter
    def agents_by_position(self, new_positions: Dict[int, int]):
        """Set the positions of the agents. This will update the agent data."""
        for position in new_positions:
            self.agent_data[new_positions[position]].position = position

    @property
    def progress(self):
        """The current progress of the agents. This is a dict of progress, keyed by agent index."""
        # Assemble progress from the agent data
        return {agent: self.agent_data[agent].progress for agent in self.agent_data}

    @progress.setter
    def progress(self, new_progress: Dict[int, Union[int, float]]):
        """Set the progress of the agents. This will update the agent data."""
        for agent in new_progress:
            self.agent_data[agent].progress = new_progress[agent]



class FOneStrategyState(pyspiel.State):
    """The state of the game at a given time step."""

    def __init__(self, game, num_agents: int, total_laps: int):
        super().__init__(game)
        self.num_agents = num_agents
        self.lap = 1
        self.total_laps = total_laps
        self.agents = AgentSet(num_agents, agent_names=None)

    def _legal_actions(self, player):
        """Returns a list of legal actions, sorted in ascending order."""
        return [action.value for action in ActionEnum]
    
    def __str__(self):
        return self.serialize()
    
    def serialize(self):
        """Convert the state to a string representation."""
        base_str = f"""Lap: {self.lap}/{self.total_laps}"""
        agent_str_template = """\nAgent {}: {} @ {:.2f} life, {} position, {:.2f} progress"""
        agent_strs = [
            agent_str_template.format(
                agent.agent_id,
                agent.tire_type.to_string(),
                agent.tire_life,
                agent.position,
                agent.progress,
            )
            for agent in self.agents.agent_data.values()
        ]
        return base_str + "".join(agent_strs)
    
    def from_string(state_str: str):
        """Load the state from a previously saved string representation."""
        self = FOneStrategyState(num_agents=0, total_laps=0)
        # Split the string into lines
        lines = state_str.split("\n")
        # Parse the lap number
        self.lap = int(lines[0].split(":")[1].split("/")[0])
        self.total_laps = int(lines[0].split(":")[1].split("/")[1])
        # Parse the agent data
        agent_data = dict()
        for line in lines[1:]:
            if line:
                # Parse the line in the format "Agent {}: {} @ {:.2f} life, {} position, {:.2f} progress"
                parts = line.split(":")
                agent_id = int(parts[0].split(" ")[1])
                parts = parts[1].split(",")
                tire_type = TireType.from_string(parts[0].split("@")[0].strip())
                tire_life = float(parts[0].split("@")[1].strip().split(" ")[0])
                position = int(parts[1].strip().split(" ")[0])
                progress = float(parts[2].strip().split(" ")[0])
                agent_data[agent_id] = AgentData(
                    agent_id=agent_id,
                    tire_type=tire_type,
                    tire_life=tire_life,
                    position=position,
                    progress=progress,
                )
        self.num_agents = len(agent_data)
        self.agents = AgentSet(len(agent_data), None)
        self.agents.agent_data = agent_data

    def action_to_string(self, agent_idx: int, action: int):
        action_enum = ActionEnum(action)
        return f"Agent {action_enum.to_string()}"
    
    def _calculate_progress_derivatives(self):
        """Calculate the progress derivatives for each agent."""
        d_progress = [0 for _ in range(self.num_agents)]
        for agent in range(self.num_agents):
            if self.agents[agent].tire_life <= 0:
                d_progress[agent] = 0
            else:
                d_progress[agent] = BASE_LAP_RELATIVE_PROGRESS + RELATIVE_PROGRESS_MODIFIER[self.agents[agent].current_move.name]
        return d_progress
    
    def _eval_overtake_manoeuvre(self, attacking_agent: AgentData, defending_agent: AgentData, forecast_progress: list) -> bool:
        """
        This function will evaluate if an overtake maneuver is successful, and then execute it if it is.
        The boolean will tell us if the overtake was successful.
        """
        if (
            attacking_agent.current_move is ActionEnum.PIT
            or defending_agent.current_move is ActionEnum.PIT
        ):
            raise ValueError("This method needs validated input")
        overtake_executed = False
        if (
            forecast_progress[attacking_agent.agent_id]
            > forecast_progress[defending_agent.agent_id]
        ):
            if np.random.random() <= overtake_probability(
                attacking_agent, defending_agent
            ):
                attacking_agent.position, defending_agent.position = (
                    defending_agent.position,
                    attacking_agent.position,
                )
                overtake_executed = True
            else:
                forecast_progress[attacking_agent.agent_id] = forecast_progress[
                    defending_agent.agent_id
                ]
        return overtake_executed
    
    def _calculate_new_track_agent_state(
        self, active_agents: List[int], forecast_progress: List[float]
    ):
        # We initialize a new dictionary to store if an agent has moved already or not
        moved = {agent: False for agent in active_agents}
        # We mask the progress array to only include active agents
        active_agent_set = self.agents.get_sliced_copy(active_agents)
        # Now we resolve any/all conflicts based on the rules of the game
        # Rule 1) If an agent is pitting, they are exempt from the conflict resolution
        # Rule 2) If an agent is not pitting, they are not allowed to move more than one position
        # Rule 3) Overtakes are resolved by comparing the overtake probability
        for position, defending_agent_idx in list(
            active_agent_set.agents_by_position.items()
        )[:-1]:
            # We move through the positions from 0 to n-1, where n is the number of active agents
            # We check each agent for a move. A move is warranted if the agent is not pitting, has not moved yet,
            # and the forecast progress of the attacking agent is greater than or equal to the defending agent.
            defending_agent = active_agent_set[defending_agent_idx]
            attacking_agent_idx = active_agent_set.agents_by_position[position + 1]
            attacking_agent = active_agent_set[attacking_agent_idx]
            if not moved[defending_agent_idx] and not moved[attacking_agent_idx]:
                overtake_executed = self._eval_overtake_manoeuvre(
                    attacking_agent, defending_agent, forecast_progress
                )
                if overtake_executed:
                    moved[attacking_agent_idx] = True
                    moved[defending_agent_idx] = True
        for agent in active_agents:
            active_agent_set[agent].progress = forecast_progress[agent]

        return active_agent_set

    def _merge_active_agents(
        self, active_agent_set: AgentSet, inactive_agent_set: AgentSet
    ):
        # We need to merge the active agents with the inactive agents
        # We assume that the progress of both sets are already updated
        for agent_id in range(self.num_agents):
            if agent_id in active_agent_set:
                self.agents[agent_id] = active_agent_set[agent_id]
            else:
                self.agents[agent_id] = inactive_agent_set[agent_id]
        # We need to update the positions of the agents in the new state
        # agents_by_progress = np.argsort([self.agent_set[agent].progress for agent in self.agents])[::-1]
        progress_values_by_agent = [
            self.agents[agent].progress for agent in range(self.num_agents)
        ]
        progress_levels = np.sort(np.unique(progress_values_by_agent))[::-1]

        # The list is from greatest progress to least progress.
        # Note that we know already that the progress order and the agent order are congruent.
        # We only need to resolve conflicts at a given progress levelset
        current_position = 0
        for progress_level in progress_levels:
            agents_at_level = np.where(progress_values_by_agent == progress_level)[0]
            # We need a sublist of the active and inactive agents at this progress level.
            active_agents_at_level = [
                agent for agent in agents_at_level if agent in active_agent_set
            ]
            inactive_agents_at_level = [
                agent for agent in agents_at_level if agent in inactive_agent_set
            ]
            # We know that the active agents are already sorted by position
            active_agent_positions = [
                active_agent_set[agent].position for agent in active_agents_at_level
            ]
            sorted_agent_internal_idxs = np.argsort(active_agent_positions)
            for idx in sorted_agent_internal_idxs:
                agent_idx = active_agents_at_level[idx]
                self.agents[agent_idx].position = current_position
                current_position += 1
            for agent_idx in inactive_agents_at_level:
                self.agents[agent_idx].position = current_position
                current_position += 1

    def calculate_new_positions_and_progress(
        self,
    ) -> Tuple[Dict[int, int], Dict[int, Union[int, float]]]:
        # We initialize a new dictionary to store if an agent has moved already or not
        d_progress = self._calculate_progress_derivatives()
        forecast_progress = self.agents.progress.copy()
        for agent in range(self.num_agents):
            forecast_progress[agent] += d_progress[agent]
        pit_mask = np.array(
            [
                self.agents[agent].current_move == ActionEnum.PIT
                for agent in range(self.num_agents)
            ],
            dtype=bool,
        )
        inactive_mask = pit_mask
        active_agents = np.array([i for i in range(self.num_agents)])[~inactive_mask]
        inactive_agents = np.array([i for i in range(self.num_agents)])[inactive_mask]
        # We calculate the new positions of the active agents
        active_agent_set = self._calculate_new_track_agent_state(
            active_agents, forecast_progress
        )
        # We calculate the new progress of the inactive agents
        inactive_agent_set = self.agents.get_sliced_copy(inactive_agents)
        for agent in inactive_agents:
            inactive_agent_set[agent].progress = forecast_progress[agent]
        # We merge the active and inactive agents
        self._merge_active_agents(active_agent_set, inactive_agent_set)

        return self.agents.positions_by_agent, self.agents.progress

    def apply_actions(self, actions: List[int]):
        for agent, action in enumerate(actions):
            self.agents[agent].current_move = ActionEnum(action)

        new_positions, new_progress = self.calculate_new_positions_and_progress()

        # Update tire degradation
        for agent in range(self.num_agents):
            self.agents[agent].tire_life = tire_degradation(self.agents[agent])

        self.lap += 1

    @property
    def _race_complete(self):
        return True if self.lap >= self.total_laps else False

    def chance_outcomes(self):
        [(0, 1.0)]  # The game is Stochastic Sampled, so only a dummy returns

    def current_player(self):
        if self._race_complete:
            return pyspiel.PlayerId.TERMINAL
        else:
            return pyspiel.PlayerId.SIMULTANEOUS
    
    def is_chance_node(self):
        return False
    
    def is_simultaneous_node(self):
        if not self._race_complete:
            return True
        return False
    
    def is_terminal(self):
        return self._race_complete
    
    # def observation_string(self, player_id: int):
    #     return f"{player_id}:{self.serialize()}"  # All players get full observability.
    
    def observation_tensor(self, player_id: int):
        """
        Here we need to return a list of floats that represent the full state. The data schema is presented below:
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
        if player_id < 0:
            raise RuntimeError(f"player >= 0")
        elif player_id >= self.num_agents:
            raise RuntimeError("player <")
        obs = [self.lap, self.total_laps, player_id]
        position_list = [self.agents[agent].position for agent in range(len(self.agents))]
        progress_list = [self.agents[agent].progress for agent in range(len(self.agents))]
        tire_life_list = [self.agents[agent].tire_life for agent in range(len(self.agents))]
        tire_type_list = [self.agents[agent].tire_type for agent in range(len(self.agents))]
        action_list = [self.agents[agent].current_move for agent in range(len(self.agents))]
        obs.extend(position_list)
        obs.extend(progress_list)
        obs.extend(tire_life_list)
        obs.extend(tire_type_list)
        obs.extend(action_list)
        obs = np.array(obs, dtype=np.float32)
        # print("Debug Obs:",obs)
        return obs
    
    # def observation_tensor(self):
    #     """
    #     Here we need to return a list of floats that represent the full state. The data schema is presented below:
    #     The Tire Type is One Hot Encoded - Currently only one tire type is available
    #     - 0: Current Lap
    #     - 1: Total Laps
    #     - 2-2+num_agents: Position of other agents by agent ID
    #     - 2+num_agents-2+2*num_agents: Progress of other agents by agent ID
    #     - 2+2*num_agents-2+3*num_agents: Tire Life of other agents by agent ID
    #     - 2+3*num_agents-2+4*num_agents: Tire Type of other agents by agent ID
    #     - 2+4*num_agents-2+5*num_agents: Last Action of other agents by agent ID
    #     """
    #     obs = [self.lap, self.total_laps]
    #     progress_list = [self.agents[agent].progress for agent in len(self.agents)]
    #     tire_life_list = [self.agents[agent].tire_life for agent in len(self.agents)]
    #     tire_type_list = [self.agents[agent].tire_type for agent in len(self.agents)]
    #     action_list = [self.agents[agent].current_move for agent in len(self.agents)]
    #     obs.extend(progress_list)
    #     obs.extend(tire_life_list)
    #     obs.extend(tire_type_list)
    #     obs.extend(action_list)
    #     obs = np.array(obs, dtype=np.float32)
    #     return obs
    
    def information_state_tensor(self, player_id: int):
        if player_id < 0:
            raise RuntimeError(f"player >= 0")
        elif player_id >= self.num_agents:
            raise RuntimeError("player <")
        return self.observation_tensor(player_id)
        
    def information_state_string(self, player_id: int):
        if player_id < 0:
            raise RuntimeError(f"player >= 0")
        elif player_id >= self.num_agents:
            raise RuntimeError("player <")
        return self.observation_string(player_id)
        
    def returns(self):
        """
        Return the reward for each agent.
        The reward is the inverse of the order of the agents at the end of the race. Otherwise it is 0.
        """
        if self._race_complete:
            positions = np.array([self.agents[agent].position for agent in range(self.num_agents)])
            rewards = len(positions) - positions
            return rewards.tolist()
        else:
            return np.zeros(self.num_agents, dtype=np.float32)
        
    def rewards(self):
        return self.returns()



def overtake_probability(
    attacking_agent: AgentData, defending_agent: AgentData
) -> float:
    """

    Args:
        attacking_agent (dict): _description_
        defending_agent (dict): _description_

    Returns:
        _type_: _description_
    """
    attacking_tire_factor = (
        0.5 + (attacking_agent.tire_life / MAX_TIRE_LIFE)
    ) * TIRE_PERFORMANCE[attacking_agent.tire_type.name]
    defending_tire_factor = (
        0.5 + (defending_agent.tire_life / MAX_TIRE_LIFE)
    ) * TIRE_PERFORMANCE[defending_agent.tire_type.name]
    attacking_action_factor = OVERTAKE_PROBABILITY_MODIFIER[
        attacking_agent.current_move.name
    ]
    defending_action_factor = DEFEND_PROBABILITY_MODIFIER[
        defending_agent.current_move.name
    ]
    prob = (BASE_ATTACK_FACTOR * attacking_tire_factor * attacking_action_factor) / (
        (BASE_ATTACK_FACTOR * attacking_tire_factor * attacking_action_factor)
        + (BASE_DEFENSE_FACTOR * defending_tire_factor * defending_action_factor)
        + 1e-6
    )
    return prob

def tire_degradation(agent: AgentData) -> np.float64:
    """

    Args:
        agent (dict): _description_

    Returns:
        _type_: _description_
    """
    if agent.current_move is ActionEnum.PIT:
        return MAX_TIRE_LIFE
    else:
        wear = (
            TIRE_DEGRADATION[agent.tire_type.name]
            * TIRE_WEAR[agent.current_move.name]
            * PER_LAP_TIRE_WEAR_FACTOR
        )
        return agent.tire_life - wear