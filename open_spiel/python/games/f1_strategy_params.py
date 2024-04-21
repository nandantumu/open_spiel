import enum

class ActionEnum(enum.IntEnum):
    PIT = 0
    PUSH = 1
    BLOCK = 2
    CONSERVE = 3

    def to_string(self):
        return self.name
    
    def from_string(action_str):
        return ActionEnum[action_str]

class TireType(enum.IntEnum):
    MEDIUM = 0

    def to_string(self):
        return f"{self.name} tire"

    def from_string(tire_type_str):
        split_str = tire_type_str.split(" ")
        if len(split_str) != 2:
            raise ValueError(f"Invalid tire type string: {tire_type_str}")
        if split_str[1] != "tire":
            raise ValueError(f"Invalid tire type string: {tire_type_str}")
        return TireType[split_str[0]]

TIRE_TYPES = ["MEDIUM"]
TIRE_TYPE_DISPLAY_ABBREVIATIONS = {"MEDIUM": "M"}
TIRE_PERFORMANCE = {"MEDIUM": 1.0}
MAX_TIRE_LIFE = 100
TIRE_DEGRADATION = {"MEDIUM": 1.0}
TIRE_WEAR = {
    "PIT": "RESET",
    "PUSH": 3,
    "BLOCK": 2,
    "CONSERVE": 1,
}
PER_LAP_TIRE_WEAR_FACTOR = 5
PIT_POSITION_PENALTY:int = 2
OVERTAKE_PROBABILITY_MODIFIER = {
    "PIT": 1.0,  # The pitting agent has already moved
    "PUSH": 2,
    "BLOCK": 0.1,
    "CONSERVE": 0.25,
}
DEFEND_PROBABILITY_MODIFIER = {
    "PIT": 1.0,  # The pitting agent has already moved
    "PUSH": 2,
    "BLOCK": 2,
    "CONSERVE": 0.25,
}
RELATIVE_PROGRESS_MODIFIER = {
    "PIT": .75,
    "PUSH": 1.05,
    "BLOCK": 0.95,
    "CONSERVE": 0.9,
}
BASE_LAP_RELATIVE_PROGRESS = 100.
BASE_ATTACK_FACTOR = 50.
BASE_DEFENSE_FACTOR = 75.