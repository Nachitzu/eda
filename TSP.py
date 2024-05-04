import numpy as np
import random
from copy import deepcopy
from enum import Enum
from itertools import combinations

class QAP_Instance:
  def __init__(self, flow, distance):
    self.flow_matrix = flow
    self.distance_matrix = distance

class QAP_State:
  def __init__(self, instance, assignment_array = [0]):
    self.instance = instance
    self.assignment_array = assignment_array
    self.not_assigned_array = set(range(len(self.instance.flow_matrix))) - set(self.assignment_array)
    self.all_assigned = len(self.not_assigned_array) == 0
    self.cost = 0
    self.update_cost()

  def calculate_cost(self, location1, location2):
    facility1 = self.assignment_array[location1]
    facility2 = self.assignment_array[location2]
    return self.instance.flow_matrix[facility1][facility2] * \
            self.instance.distance_matrix[location1][location2]

  def update_cost(self):
    n = len(self.assignment_array)
    if n > 1:
      cost = 0
      for i in range(n):
        for j in range(n):
          cost += self.calculate_cost(i, j)
      self.cost = cost

  def __deepcopy__(self, memo):
    new_instance = type(self) (
        instance = self.instance,
        assignment_array = deepcopy(self.assignment_array)
    )
    return new_instance

  def __str__(self):
    return f"Asignación : {self.assignment_array}\nCosto      : {self.cost}"

class ActionType(Enum):
  CONSTRUCTIVE = 1
  SWAP = 2
  ROTATION = 3
  SHIFT = 4

class QAP_Environment():
  @staticmethod
  def gen_actions(state, action_type, shuffle = False):

    if action_type == ActionType.CONSTRUCTIVE:
      actions = [(action_type, facility) for facility in state.not_assigned_array]

    elif action_type == ActionType.SWAP:
      n = len(state.assignment_array)
      actions = [(action_type, (i, j)) for i in range(n - 1) for j in range(i + 1, n)]

    elif action_type == ActionType.SHIFT:
      n = len(state.assignment_array)
      actions = []
      for i in range(n):
        for j in range(n):
          if i != j:
            actions.append((action_type, (i, j)))

    elif action_type == ActionType.ROTATION:
      n = len(state.assignment_array)
      k = 3
      actions = []
      subsets = combinations(range(n), k)
      for subset in subsets:
        actions.append((action_type, (subset, 'left')))
        actions.append((action_type, (subset, 'right')))

    else:
      raise NotImplementedError(f"Tipo de acción '{type}' no implementado")

    if shuffle:
      random.shuffle(actions)

    for action in actions:
      yield action

  @staticmethod
  def state_transition(state, action):
    # constructive-move: Agrega una instalacion a las asignaciones.
    if action[0] == ActionType.CONSTRUCTIVE and state.all_assigned == False:
      state.assignment_array.append(action[1])
      state.not_assigned_array.remove(action[1])

      if len(state.not_assigned_array) == 0:
        state.update_cost()
        state.all_assigned = True

    # swap: Intercambia dos instalaciones de ubicacion.
    elif action[0] == ActionType.SWAP and state.all_assigned == True:
      i, j = action[1]
      state.assignment_array[i], state.assignment_array[j] = \
        state.assignment_array[j], state.assignment_array[i]
      state.update_cost()

    elif action[0] == ActionType.SHIFT and state.all_assigned == True:
      i, j = action[1]
      elem = state.assignment_array.pop(i)
      state.assignment_array.insert(j, elem)
      state.update_cost()

    elif action[0] == ActionType.ROTATION and state.all_assigned == True:
      subset, direction = action[1]

      n = len(state.assignment_array)
      k = len(subset)

      # Obtener el subconjunto correspondiente del arreglo original
      subset_arr = [state.assignment_array[i] for i in subset]

      # Aplicar la rotación
      if direction == "right":
          rotated_subset = subset_arr[-1:] + subset_arr[:-1]
      elif direction == "left":
          rotated_subset = subset_arr[1:] + subset_arr[:1]
      else:
          raise ValueError("La dirección debe ser 'right' o 'left'")

      for i, j in zip(subset, rotated_subset):
          state.assignment_array[i] = j

      state.update_cost()

    else:
      raise NotImplementedError(f"Movimiento '{action}' no válido para estado {state}, all asigned: {state.all_assigned}, not assigned: {state.not_assigned_array}")

    return state

  @staticmethod
  def calculate_cost_after_action(state, action): # TODO: Mejorar algoritmo
    if action[0] == ActionType.SWAP:
      k, l = action[1]

      state.assignment_array[k], state.assignment_array[l] = \
        state.assignment_array[l], state.assignment_array[k]

      new_cost = 0
      n = len(state.assignment_array)
      for i in range(n):
        for j in range(n):
          facility1 = state.assignment_array[i]
          facility2 = state.assignment_array[j]
          location1 = i
          location2 = j
          new_cost += state.instance.flow_matrix[facility1][facility2] * \
                      state.instance.distance_matrix[location1][location2]

      state.assignment_array[l], state.assignment_array[k] = \
        state.assignment_array[k], state.assignment_array[l]

      return new_cost

    return state.cost
  
#@handle_multiple_states
def evalConstructiveActions(state, env, debug):
  evals = []

  facility1 = state.assignment_array[-1]
  n = len(state.assignment_array)

  location1 = n - 1
  location2 = n

  for action in env.gen_actions(state, ActionType.CONSTRUCTIVE):
    facility2 = action[1]

    if facility1 == facility2:
      continue;

    if debug:
      print(f"Facilites: {facility1}, {facility2}")
      print(f"Locations: {location1}, {location2}")

    cost = state.instance.flow_matrix[facility1][facility2] * \
           state.instance.distance_matrix[location1][location2]

    if debug:
      print(f"Cost: {cost} | Flow: {state.instance.flow_matrix[facility1][facility2]} | Distance: {state.instance.distance_matrix[location1][location2]}")

    evals.append((action, -cost))

  if debug and len(evals) > 0:
    print("Actions : ", evals)

  return evals

import matplotlib.pyplot as plt

def plot_tour(points, visited, start_node=False):
    # Asegurarse de que 'visited' contiene índices válidos para 'points'
    if not all(0 <= i < len(points) for i in visited):
        raise ValueError("Los índices en 'visited' deben ser válidos para 'points'")

    # Separar las coordenadas x e y de los puntos
    x = [points[i][0] for i in visited]
    y = [points[i][1] for i in visited]

    # Agregar el primer punto al final para cerrar el tour
    if start_node==False:
      x.append(x[0])
      y.append(y[0])

    # Graficar los puntos
    plt.scatter(x, y)

    # Graficar las líneas del tour
    plt.plot(x, y)

    # Agregar títulos y etiquetas si es necesario
    plt.title("Tour de puntos 2D")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")

    # Mostrar el gráfico
    plt.show()