import game
import numpy as np

g = game.Game("dcqn_agent", "cuda")


hyperparameter = {
  "lr_start": 1e-4,
  "lr_end": 1e-4,
  "batch_size": 64,
  "gamma": 0.9,
  "eps_start": 0.9,
  "eps_end": 1e-2
}


g.train_agent(False, 200, 32, hyperparameter)  
score = g.main(True)
print("Score: ", score)
