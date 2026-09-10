from langgraph.graph import StateGraph, END
from rl_with_langgraph.QLearning_agent import QLearningAgent
from rl_with_langgraph.customer import Customer

class NegotiationGame:
    ACTIONS = ["hold", "small_concede", "big_concede", "accept", "walk_away"]
    MAX_ROUNDS = 6
    COST_FLOOR = 60
    WALK_PENALTY = -5
    TIMEOUT_PENALTY = 0
    STEP_COST = -0.5

    def __init__(self, agent: QLearningAgent):
        self.agent = agent
        self.customer = None
        self.graph = self.build_graph()

    def build_graph(self):
        g = StateGraph(dict)
        g.add_node("agent_move", self.agent_move)
        g.add_node("customer_move", self.customer_move)
        g.add_node("learn", self.learn)
        g.set_entry_point("agent_move")
        g.add_edge("agent_move", "customer_move")
        g.add_edge("customer_move", "learn")
        g.add_conditional_edges("learn", lambda s: END if s["done"] else "agent_move")
        return g.compile()

    def agent_move(self, state):
        s = self.agent.get_state(state["round_num"], state["agent_offer"], state["customer_offer"])
        a = self.agent.choose_action(s)
        action = self.ACTIONS[a]

        if action == "small_concede":
            state["agent_offer"] -= 5
        elif action == "big_concede":
            state["agent_offer"] -= 15
        elif action == "accept":
            state["done"] = True
            state["outcome"] = "deal"
            state["deal_price"] = state["customer_offer"]
        elif action == "walk_away":
            state["done"] = True
            state["outcome"] = "walked"

        state["last_state"] = s
        state["last_action"] = a
        return state

    def customer_move(self, state):
        if state["done"]:
            return state
        state["customer_offer"] = self.customer.make_offer(
            state["round_num"], state["agent_offer"], state["customer_offer"]
        )
        if state["agent_offer"] <= state["customer_offer"]:
            state["done"] = True
            state["outcome"] = "deal"
            state["deal_price"] = state["agent_offer"]
        return state

    def learn(self, state):
        state["round_num"] += 1
        if not state["done"] and state["round_num"] >= self.MAX_ROUNDS:
            state["done"] = True
            state["outcome"] = "timeout"

        state["reward"] = self.reward(state)
        state["episode_return"] += state["reward"]   # accumulate every round's reward

        next_state_key = self.agent.get_state(
            state["round_num"], state["agent_offer"], state["customer_offer"]
        )
        self.agent.update(
            state["last_state"], state["last_action"], state["reward"], next_state_key, state["done"]
        )
        return state

    def reward(self, state):
        outcome = state["outcome"]
        if outcome == "deal":
            return state["deal_price"] - self.COST_FLOOR
        if outcome == "walked":
            return self.WALK_PENALTY
        if outcome == "timeout":
            return self.TIMEOUT_PENALTY
        return self.STEP_COST

    def run_episode(self, customer: Customer):
        self.customer = customer
        state = {
            "round_num": 0,
            "agent_offer": 100,
            "customer_offer": 60,
            "done": False,
            "outcome": "",
            "reward": 0.0,
            "episode_return": 0.0,
            "deal_price": None,
            "last_state": None,
            "last_action": None,
        }
        result = self.graph.invoke(state)
        self.agent.decay_epsilon()
        return result, result["episode_return"]