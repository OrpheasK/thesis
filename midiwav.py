from timidity import Parser, play_notes
import numpy as np

ps = Parser("C:/Users/Papias/Desktop/thesis/copy/midi/test2.mid")

play_notes(*ps.parse(), np.sin)