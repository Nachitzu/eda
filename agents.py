class GreedyAgent():
  def __init__(self, eval_actions, debug=False):
    self.eval_actions = eval_actions
    self.debug = debug

  def reset(self):
    pass

  def select_action(self, evals):
    return max(evals, key=lambda x: x[1])[0]

  def action_policy(self, state, env):
    evals = self.eval_actions(state, env, self.debug)

    if len(evals) == 0:
      return None

    # Seleccionar la acción que maximiza su evaluación
    return self.select_action(evals)

  def __deepcopy__(self, memo):
    # Crear una nueva instancia de la clase
    new_instance = type(self) (
        eval_actions = self.eval_actions, # pasamos por referencia
        debug = self.debug
    )
    return new_instance
  
class LocalSearchAgent(GreedyAgent):
  def __init__(self, action_type, first_improvement=True, debug=False):
    self.action_type = action_type
    self.first_improvement = first_improvement
    self.debug = debug

  def eval_actions(self, state, env, debug):
    current_cost = state.cost
    evals = []

    for action in env.gen_actions(state, self.action_type, shuffle=True):
      if debug:
        print(f"Action: {action}")

      new_cost = env.calculate_cost_after_action(state, action)

      if debug:
        print(f"New Cost: {new_cost}")

      if new_cost < current_cost:
        evals.append((action, new_cost))
        if self.first_improvement:
          return evals

    return evals

def __deepcopy__(self, memo):
    # Crear una nueva instancia de la clase
    new_instance = type(self) (
      action_type = self.action_type,
      first_improvement = self.first_improvement,
      debug = self.debug
    )
    return new_instance
  
class SingleAgentSolver():
  def __init__(self, env, agent):
    self.env = env
    self.agent = agent

  def solve(self, state, track_best_state=False, save_history=False, max_actions=0):
    history = None

    if save_history:
      history = [(None, state.cost)]

    if max_actions == 0:
      max_actions = 99999999

    best_state = None

    if track_best_state:
      best_state = deepcopy(state)

    self.agent.reset()

    n_actions = 0
    while n_actions < max_actions:
      action = self.agent.action_policy(state, self.env)
      if action is None:
        break

      state = self.env.state_transition(state, action)
      n_actions += 1

      if track_best_state and state.cost < best_state.cost:
        best_state = deepcopy(state)

      if save_history:
        history.append((action, state.cost))

    if track_best_state:
      return best_state, history
    else:
      return state, history, n_actions

  def multistate_solve(self, states, track_best_state=False, save_history=False, max_actions=0):
    agents = [deepcopy(self.agent) for _ in range(len(states))]
    history = [None]*len(states)
    best_state = [None]*len(states)
    n_actions = [None]*len(states)

    if max_actions == 0:
      max_actions = 99999999

    for i in range(len(states)):
      agents[i].reset()
      n_actions[i] = 0
      history[i] = []
      if track_best_state: best_state[i] = deepcopy(states[i])

    live_states_idx = list(range(len(states)))

    for _ in range(max_actions):
        evals = agents[0].eval_actions([states[i] for i in live_states_idx], self.env, agents[0].debug)

        new_idx = []
        for i in live_states_idx:
          eval = evals[live_states_idx.index(i)]

          if eval == []:
            continue

          action = agents[i].select_action(eval)

          states[i] = self.env.state_transition(states[i], action)
          n_actions[i] += 1

          new_idx.append(i)

          if track_best_state and states[i].cost < best_state.cost:
            best_state[i] = deepcopy(states[i])

          if save_history:
            history[i].append((action, states[i].cost))

        live_states_idx = new_idx

        if new_idx == []:
          break

    if track_best_state:
      return best_state, history
    else:
      return states, history, n_actions