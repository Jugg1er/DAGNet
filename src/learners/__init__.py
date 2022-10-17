from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .mgan_learner import MganLeaner
from .g2a_learner import G2ALeaner
from .ng2a_learner import newG2ALeaner
from .mdg_learner import MDGLeaner


REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["mgan_learner"] = MganLeaner
REGISTRY["g2a_learner"] = G2ALeaner
REGISTRY["ng2a_learner"] = newG2ALeaner
REGISTRY["mdg_learner"] = MDGLeaner