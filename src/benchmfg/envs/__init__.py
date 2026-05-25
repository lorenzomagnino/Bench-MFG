"""
Environments module for Mean Field Games.

This module contains various MFG environment implementations.
"""

from .contraction_game.contraction_game import ContractionGame
from .four_rooms_obstacles.four_rooms_obstacles import FourRoomsAversion2D
from .kinetic_congestion.kinetic_congestion import KineticCongestion
from .lasry_lions_chain.lasry_lions_chain import LasryLionsChain
from .mf_garnet.mf_garnet import MFGarnet
from .mfg_model_class import MFGStationary
from .multiple_equilibria.multiple_equilibria import MultipleEquilibria1DGame
from .no_interaction.no_interaction import NoInteractionChain
from .rock_paper_scissors.rock_paper_scissors import RockPaperScissors
from .sis_epidemic.sis_epidemic import SISEpidemic

__all__ = [
    "MFGStationary",
    "LasryLionsChain",
    "NoInteractionChain",
    "FourRoomsAversion2D",
    "MultipleEquilibria1DGame",
    "MFGarnet",
    "RockPaperScissors",
    "SISEpidemic",
    "KineticCongestion",
    "ContractionGame",
]
