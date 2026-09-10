from rl_with_langgraph.training import Trainer
from rl_with_langgraph.testing import Tester
# Training
trainer = Trainer()
trainer.train()
trainer.plot_all()

# Testing
tester = Tester(trainer.agent)
tester.test()
tester.plot_all()
