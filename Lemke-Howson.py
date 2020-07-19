'''
[0,3,0]
[0,0,3]
[2,2,2]

[0,0,2]
[3,0,2]
[0,3,2]
'''
import time
import numpy as np
from itertools import cycle
u1 = np.array([[0,3,0],[0,0,3],[2,2,2]]) #player1's utility matrix
u2 = u1.T #player2's utility matrix
start_time = time.time()
'''
game = nash.Game(u1,u2) 
equilibria = game.lemke_howson_enumeration()
for eq in equilibria:
    print(eq)
'''

def polytope_tableau(M): #給定matrix A, 定義Ａz<=1, z>=0的polytope
    return np.append(
        np.append(M, np.eye(M.shape[0]), axis=1),
        np.ones((M.shape[0], 1)),
        axis=1)

def find_pivot_row(tableau, column_index): 
    return np.argmax(tableau[:, column_index] / tableau[:, -1])

def non_basic_variables(tableau):
    columns = tableau[:, :-1].transpose()
    return set(np.where([np.count_nonzero(col) != 1 for col in columns])[0])

def pivot_tableau(tableau, column_index):
    original_labels = non_basic_variables(tableau)
    pivot_row_index = find_pivot_row(tableau, column_index)
    pivot_element = tableau[pivot_row_index, column_index]

    for i, _ in enumerate(tableau):
        if i != pivot_row_index:
            tableau[i, :] = (
                tableau[i, :] * pivot_element
                - tableau[pivot_row_index, :] * tableau[i, column_index]
            )
    return non_basic_variables(tableau) - original_labels #returns the dropped label

def shift_tableau(tableau, shape): #Shift a tableau to ensure labels of pairs of tableaux coincide
    return np.append(
        np.roll(tableau[:, :-1], shape[0], axis=1),
        np.ones((shape[0], 1)),
        axis=1)

def tableau_to_strategy(tableau, basic_labels, strategy_labels):
    vertex = []
    for column in strategy_labels:
        if column in basic_labels:
            for i, row in enumerate(tableau[:, column]):
                if row != 0:
                    vertex.append(tableau[i, -1] / row)
        else:
            vertex.append(0)
    strategy = np.array(vertex)
    return strategy / sum(strategy)


def lemke_howson(U1, U2, initial_dropped_label=0):
    """
    1. Start at the artificial equilibrium (which is fully labeled)
    2. Choose an initial label to drop and move in the polytope for which
       the vertex has that label to the edge
       that does not share that label. (This is implemented using integer
       pivoting)
    3. A label will now be duplicated in the other polytope, drop it in a
       similar way.
    4. Repeat steps 2 and 3 until have Nash Equilibrium.
    """

    if np.min(U1) <= 0:
        U1 = U1 + abs(np.min(U1)) + 1
    if np.min(U2) <= 0:
        U2 = U2 + abs(np.min(U2)) + 1

    # build tableaux
    col_tableau = polytope_tableau(U1)
    col_tableau = shift_tableau(col_tableau, U1.shape)
    row_tableau = polytope_tableau(U2.transpose())
    full_labels = set(range(sum(U1.shape)))

    if initial_dropped_label in non_basic_variables(row_tableau):
        tableux = cycle((row_tableau, col_tableau))
    else:
        tableux = cycle((col_tableau, row_tableau))

    # First pivot (to drop a label)
    entering_label = pivot_tableau(next(tableux), initial_dropped_label)
    while (
        non_basic_variables(row_tableau).union(non_basic_variables(col_tableau))
        != full_labels
    ):
        entering_label = pivot_tableau(
            next(tableux), next(iter(entering_label))
        )

    row_strategy = tableau_to_strategy(
        row_tableau, non_basic_variables(col_tableau), range(U1.shape[0])
    )
    col_strategy = tableau_to_strategy(
        col_tableau,
        non_basic_variables(row_tableau),
        range(U1.shape[0], sum(U1.shape)),
    )
    return row_strategy, col_strategy

payoff_matrices = tuple([u1,u2])
equilibria=[]
for i in range(0, sum(u1.shape)):
    strategy = lemke_howson(*payoff_matrices, initial_dropped_label=i)
    equilibria.append(strategy)
for eq in equilibria:
    print(eq)
print("--- running time:%s seconds ---" % (time.time() - start_time))