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
from itertools import chain, combinations
u1 = np.array([[0,3,0],[0,0,3],[2,2,2]]) #player1's utility matrix
u2 = u1.T #player2's utility matrix
start_time = time.time()

#find pure NE
BRfor1 = np.argmax(u1, axis=0)
BRfor2 = np.argmax(u2, axis=1)

#try all support pairs
support = BRfor1 #[0 1 2]
#normal mixed NE
#p = [p0, p1, p2]
#p0+p1+p2=1
#p0*u2[0][0] + p1*u2[1][0] + p2*u2[2][0] = p0*u2[0][1] + p1*u2[1][1] + p2*u2[2][1]
#p0*u2[0][1] + p1*u2[1][1] + p2*u2[2][1] = p0*u2[0][2] + p1*u2[1][2] + p2*u2[2][2]
'''
a1_0 = u2[0][0]-u2[0][1]
a1_1 = u2[1][0]-u2[1][1]
a1_2 = u2[2][0]-u2[2][1]
a2_0 = u2[0][1]-u2[0][2]
a2_1 = u2[1][1]-u2[1][2]
a2_2 = u2[2][1]-u2[2][2]

a = np.array([[1,1,1], [a1_0,a1_1,a1_2], [a2_0,a2_1,a2_2]])
b = np.array([1,0,0])
x = np.linalg.solve(a,b)
'''
#p = [p0, p1, p2]
#p0 + p1 + p2 = 1
#for player1 support:[0]
#p0=1
#q0*u[0][0] + q1*u[0][1] + q2*u[0][2] > q0*u[1][0] + q1*u[1][1] + q2*u[1][2]
#q0*u[0][0] + q1*u[0][1] + q2*u[0][2] > q0*u[2][0] + q1*u[2][1] + q2*u[2][2]
#for player1 support:[1]
#for player1 support:[2]
#for player1 support:[0,1]
#for player1 support:[1,2]
#for player1 support:[0,2]
#for player1 support:[0,1,2]


def powerset(n): #找range(n)的所有子集合
    return chain.from_iterable(combinations(range(n), r) for r in range(n + 1))

def indifference(U1, rows=None, columns=None):
    #找出能讓player‘s payoff indifference的每個support機率
    #U1: payoff matrix for the row player)
    #rows: the support played by the row player
    #columns: the support player by the column player
    
    M = (U1[np.array(rows)] - np.roll(U1[np.array(rows)], 1, axis=0))[:-1]
    zero_columns = set(range(U1.shape[1])) - set(columns) #機率會是0的column

    if zero_columns != set():
        M = np.append(
            M,
            [[int(i == j) for i, col in enumerate(M.T)] for j in zero_columns],
            axis=0,
        )

    M = np.append(M, np.ones((1, M.shape[1])), axis=0)
    b = np.append(np.zeros(len(M) - 1), [1])

    try:
        prob = np.linalg.solve(M, b)
        if all(prob >= 0):
            return prob #return能使row player的payoff indifferent的column player機率分配
        return False
    except np.linalg.linalg.LinAlgError:
        return False


def support_pairs(U1, U2, non_degenerate=False):#find support pairs
    p1_num_strategies, p2_num_strategies = U1.shape
    for support1 in (s for s in powerset(p1_num_strategies) if len(s) > 0):
        for support2 in (
            s
            for s in powerset(p2_num_strategies)
            if (len(s) > 0 and not non_degenerate) or len(s) == len(support1)
        ):
            yield support1, support2


def indifference_strategies(U1, U2, non_degenerate=False, tol=10 ** -16):
    #從support pair生成的每個strategy都要indifferent
    if non_degenerate:
        tol = min(tol, 0)

    for pair in support_pairs(U1, U2, non_degenerate=non_degenerate):
        s1 = indifference(U2.T, *(pair[::-1]))
        s2 = indifference(U1, *pair)

        if obey_support(s1, pair[0], tol=tol) and obey_support(
            s2, pair[1], tol=tol
        ):
            yield s1, s2, pair[0], pair[1]


def obey_support(strategy, support, tol=10 ** -16):
    #測試策略有沒有服從support
    if strategy is False:
        return False
    if not all(
        (i in support and value > tol) or (i not in support and value <= tol)
        for i, value in enumerate(strategy)
    ):
        return False
    return True


def is_NE(strategy_pair, support_pair, payoff_matrices):
    #看輸入的策略中包含的每個元素都是best response
    U1, U2 = payoff_matrices
    # Payoff against opponents strategies:
    u = strategy_pair[1].reshape(strategy_pair[1].size, 1)
    row_payoffs = np.dot(U1, u)

    v = strategy_pair[0].reshape(strategy_pair[0].size, 1)
    column_payoffs = np.dot(U2.T, v)

    # Pure payoffs on current support:
    row_support_payoffs = row_payoffs[np.array(support_pair[0])]
    column_support_payoffs = column_payoffs[np.array(support_pair[1])]

    return (
        row_payoffs.max() == row_support_payoffs.max()
        and column_payoffs.max() == column_support_payoffs.max()
    )

def support_enumeration(U1, U2, non_degenerate=False, tol=10 ** -16):
    #列舉所有support找出NE
    count = 0
    for s1, s2, sup1, sup2 in indifference_strategies(
        U1, U2, non_degenerate=non_degenerate, tol=tol
    ):
        if is_NE((s1, s2), (sup1, sup2), (U1, U2)):
            count += 1
            if (s1==s2).all()==False:
                yield np.abs(s1), np.abs(s2)
            
payoff_matrices = tuple([u1,u2])
equilibria = support_enumeration(*payoff_matrices)
for eq in equilibria:
    print(eq)
print("--- running time:%s seconds ---" % (time.time() - start_time))