import numpy as np

class Policy:
   @staticmethod
   def policy(q_table,state,explore=0.0):
       action = np.argmax(q_table[state])
       if np.random.random() < explore:
        action = np.random.randint(low=0, high=6)
       return action