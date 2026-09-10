import random

class Customer:
    def __init__(self, strategy: str):
        self.strategy = strategy
        self.true_value = 100

    def make_offer(self,round_num,agent_offer,customer_last_offer):
        if self.strategy == "patient":
            target = 60 + round_num * 3
        elif self.strategy == "impatient":
            target = 60 + round_num * 8
        elif self.strategy == "tit_for_tat":
            target = customer_last_offer + (agent_offer - customer_last_offer) * 0.5
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return min(target, self.true_value)