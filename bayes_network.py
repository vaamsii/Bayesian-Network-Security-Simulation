
import sys

'''
WRITE YOUR CODE BELOW.
'''
from numpy import zeros, float32
#  pgmpy͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import random
import numpy as np
#You are not allowed to use following set of modules from 'pgmpy' Library.͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
#
# pgmpy.sampling.*͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
# pgmpy.factors.*͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
# pgmpy.estimators.*͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁

def make_security_system_net():
    """Create a Bayes Net representation of the above security system problem. 
    Use the following as the name attribute: "H","C", "M","B", "Q", 'K",
    "D"'. (for the tests to work.)
    """
    BayesNet = BayesianNetwork()

    # Add the nodes given in the description above also it's listed in read me
    # I add the nodes on at a time in the for loop to the bayesian network.
    nodes = ["H", "C", "M", "B", "Q", "K", "D"]
    for node in nodes:
        BayesNet.add_node(node)

    # Add edges with the children to their respective parents
    # Hacking Q's database depends on hiring hackers (H) and having Contra (C),
    # H and C are parents of Q.
    BayesNet.add_edge("H", "Q")
    BayesNet.add_edge("C", "Q")

    # Kidnapping M depends on hiring mercenaries (M) and whether Bond is guarding M (B)
    # M and B are parents of K
    BayesNet.add_edge("M", "K")
    BayesNet.add_edge("B", "K")

    # Obtaining the “Double-0” files depends on having both the cipher (Q) and the key (K)
    # Q and K are parents of D
    BayesNet.add_edge("Q", "D")
    BayesNet.add_edge("K", "D")

    return BayesNet


def set_probability(bayes_net):
    """Set probability distribution for each node in the security system.
    Use the following as the name attribute: "H","C", "M","B", "Q", 'K",
    "D"'. (for the tests to work.)
    """

    # cpd_a = TabularCPD('A', 2, values=[[0.3], [0.7]])
    # Probabilities for nodes without parents
    # Spectre hires hackers
    cpd_h = TabularCPD('H', 2, values=[[0.5], [0.5]])
    # Spectre buys Contra
    cpd_c = TabularCPD('C', 2, values=[[0.7], [0.3]])
    # Spectre hires mercenaries
    cpd_m = TabularCPD('M', 2, values=[[0.2], [0.8]])
    # Bond is guarding M
    cpd_b = TabularCPD('B', 2, values=[[0.5], [0.5]])

    # cpd_tag = TabularCPD('T', 2, values=[[0.9, 0.8, 0.4, 0.85], \
    #                                      [0.1, 0.2, 0.6, 0.15]], evidence=['A', 'G'], evidence_card=[2, 2])

    # I solved the below CPDs using truth table. Here is an example for P(Q|H,C)

    # H	        C	        P(Q=true)
    # True	    True	    0.9
    # True	    False	    0.55
    # False	    True	    0.25
    # False	    False	    0.05

    # Probabilities for nodes with parents
    # P(Q|H,C)

    # the first index in the values list is a list of the P(Q=false) second index is P(Q=true).
    # Rest of CPDs follow same process

    cpd_q = TabularCPD('Q', 2, values=[[0.95, 0.75, 0.45, 0.1],
                                       [0.05, 0.25, 0.55, 0.9]],
                       evidence=['H', 'C'], evidence_card=[2, 2])

    # P(K|M,B)
    cpd_k = TabularCPD('K', 2, values=[[0.25, 0.99, 0.05, 0.85],
                                       [0.75, 0.01, 0.95, 0.15]],
                       evidence=['M', 'B'], evidence_card=[2, 2])

    # P(D|Q,K)
    cpd_d = TabularCPD('D', 2, values=[[0.98, 0.65, 0.4, 0.01],
                                       [0.02, 0.35, 0.6, 0.99]],
                       evidence=['Q', 'K'], evidence_card=[2, 2])

    # Adding CPDs to the Bayesian Network
    bayes_net.add_cpds(cpd_h, cpd_c, cpd_m, cpd_b, cpd_q, cpd_k, cpd_d)
    return bayes_net


def get_marginal_double0(bayes_net):
    """Calculate the marginal probability that Double-0 gets compromised.
    """

    # solver = VariableElimination(bayes_net)
    # marginal_prob = solver.query(variables=['A'], joint=False)
    # prob = marginal_prob['A'].values

    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['D'], joint=False)
    double0_prob = marginal_prob['D'].values[1]
    return double0_prob


def get_conditional_double0_given_no_contra(bayes_net):
    """Calculate the conditional probability that Double-0 gets compromised
    given Contra is shut down.
    """

    # here we are computing P('A' = false | 'B' = true, 'C' = False)
    # solver = VariableElimination(bayes_net)
    # conditional_prob = solver.query(variables=['A'], evidence={'B': 1, 'C': 0}, joint=False)
    # prob = conditional_prob['A'].values

    # I used the exact similar approach as previous method except we are saying,
    # when is D true when C is false. That's the probability we are finding. Hence, why we added evidence.

    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['D'], evidence={'C': 0}, joint=False)
    double0_prob = conditional_prob['D'].values[1]
    return double0_prob


def get_conditional_double0_given_no_contra_and_bond_guarding(bayes_net):
    """Calculate the conditional probability that Double-0 gets compromised
    given Contra is shut down and Bond is reassigned to protect M.
    """

    # here we are computing P('A' = false | 'B' = true, 'C' = False)
    # solver = VariableElimination(bayes_net)
    # conditional_prob = solver.query(variables=['A'], evidence={'B': 1, 'C': 0}, joint=False)
    # prob = conditional_prob['A'].values

    solver = VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['D'], evidence={'C': 0, 'B': 1}, joint=False)
    double0_prob = conditional_prob['D'].values[1]
    return double0_prob


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """

    BayesNet = BayesianNetwork()
    # Adding the respective nodes in this bayesian network, we add each node using the for loop.
    nodes = ["A", "B", "C", "AvB", "BvC", "CvA"]
    for node in nodes:
        BayesNet.add_node(node)

    # This is my understanding of the bayesian network for this problem:
    # A -> AVB <- B
    # B -> BVC <- C
    # C -> CVA <- A
    # CVA, AVB and BVC are all children.

    # Adding edges for children with their respective parents.
    BayesNet.add_edge("A", "AvB")
    BayesNet.add_edge("B", "AvB")
    BayesNet.add_edge("B", "BvC")
    BayesNet.add_edge("C", "BvC")
    BayesNet.add_edge("C", "CvA")
    BayesNet.add_edge("A", "CvA")

    #0-> 0.15
    #1->0.45
    #2->0.30
    #3->0.10

    # I am using the following code in comments from previous part to help me assign the right skill levels to each team

    # cpd_a = TabularCPD('A', 2, values=[[0.3], [0.7]])

    # cpd_tag = TabularCPD('T', 2, values=[[0.9, 0.8, 0.4, 0.85], \
    #                                      [0.1, 0.2, 0.6, 0.15]], evidence=['A', 'G'], evidence_card=[2, 2])

    cpd_a = TabularCPD('A', 4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_b = TabularCPD('B', 4, values=[[0.15], [0.45], [0.30], [0.10]])
    cpd_c = TabularCPD('C', 4, values=[[0.15], [0.45], [0.30], [0.10]])

    # Define match outcome probabilities based on skill differences
    # Probabilities for AvB, BvC, CvA based on the skill difference table provided

    # I found the following using an truth table by hand.
    # For example, if we have A skill and B skill. Our goal is to get the probability for AvB using those two states.
    # first state is when A skill 0 and B skill is 0. so the probablity based of the second table given of A winning is 0.10 and B winning is 0.10
    # The chance of a tie is 0.80. This is when A is 0 and B is 0. So the state is (0,0). Like this I did upto 16 state calculatoions
    # 16 because A has 4 different skill levels and B has 4 different as well, which is 4*4 = 16 possible states.
    # I used those values respectability in the below list. I gave a small snippet below of few states in my truth table.
    # A Skill B Skill   A Wins (P)	B Wins (P)	Tie (P)
    # 0	      0	        0.10	        0.10	    0.80
    # 0	      1	        0.20	        0.60	    0.20
    # 0	      2	        0.15	        0.75	    0.10
    # 0	      3	        0.05	        0.90	    0.05

    match_outcome_probs = [
        # Probabilities for A wins
        [0.10, 0.20, 0.15, 0.05, 0.60, 0.10, 0.20, 0.15, 0.75, 0.60, 0.10, 0.20, 0.90, 0.75, 0.60, 0.10],
        # Probabilities for B wins
        [0.10, 0.60, 0.75, 0.90, 0.20, 0.10, 0.60, 0.75, 0.15, 0.20, 0.10, 0.60, 0.05, 0.15, 0.20, 0.10],
        # Probabilities for Tie
        [0.80, 0.20, 0.10, 0.05, 0.20, 0.80, 0.20, 0.10, 0.10, 0.20, 0.80, 0.20, 0.05, 0.10, 0.20, 0.80]
    ]

    cpd_avb = TabularCPD('AvB', 3,
                         values=match_outcome_probs,
                         evidence=['A', 'B'], evidence_card=[4, 4])
    cpd_bvc = TabularCPD('BvC', 3,
                         values=match_outcome_probs,
                         evidence=['B', 'C'], evidence_card=[4, 4])
    cpd_cva = TabularCPD('CvA', 3,
                         values=match_outcome_probs,
                         evidence=['C', 'A'], evidence_card=[4, 4])

    # we add the cpds to the bayesian network which can be used later.

    BayesNet.add_cpds(cpd_a, cpd_b, cpd_c, cpd_avb, cpd_bvc, cpd_cva)
    return BayesNet


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]

    # we are asking to find BvC probability when AvC is 0 and CvA is 2.

    # solver = VariableElimination(bayes_net)
    # conditional_prob = solver.query(variables=['A'], evidence={'B': 1, 'C': 0}, joint=False)
    # prob = conditional_prob['A'].values

    solver = VariableElimination(bayes_net)
    posterior_prob = solver.query(variables=['BvC'], evidence={'AvB': 0, 'CvA': 2})
    posterior = posterior_prob.values

    return posterior # list


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    sample = tuple(initial_state)    

    # base case, If an initial value is not given (initial state is None or and empty list
    # default to a state chosen uniformly at random from the possible states
    if initial_state is None or len(initial_state) == 0:
        initial_state = []

        # I am assigning the skill levels randomly and appending to the initial_state
        for i in range(3):
            skill_level = random.randint(0,3)
            initial_state.append(skill_level)

        # avb outcome, we already know it's 0.
        initial_state.append(0)

        bvc_outcome = random.randint(0,2)
        initial_state.append(bvc_outcome)

        # cva outcome, we already know it's 2.
        initial_state.append(2)

    # initializing the nodes for this game
    nodes = ["A", "B", "C", "AvB", "BvC", "CvA"]

    # we need to resample only following nodes, since we know outcome of AvB and CvA, then we choose randomly using random.choice
    resample_nodes = ["A", "B", "C", "BvC"]
    node_to_resample = random.choice(resample_nodes)

    # I am using hint 1 here:
    # getting all the CPDs from the bayesian network
    cpd_a = bayes_net.get_cpds("A")
    cpd_b = bayes_net.get_cpds("B")
    cpd_c = bayes_net.get_cpds("C")
    cpd_avb = bayes_net.get_cpds("AvB")
    cpd_bvc = bayes_net.get_cpds("BvC")
    cpd_cva = bayes_net.get_cpds("CvA")

    # I am trying to find the conditional probability, so initializing an empty list
    conditional_prob = []
    # I am checking if the resample node isn't BvC, which means the resample node is A,B,C
    if node_to_resample != "BvC":
        # We are looping in 4 iterations since there are 4 different skill levels possible for A,B,C
        for skill_level in range(4):
            # my logic here is if for example node to resample is A
            # We are looking to find the probability of the following:
            # P(A| B, C, AvB, BvC, CvA)
            # Essentially we can use Markov blanket approach for this
            # the rule that Markov blanket says is if the variable don't have direct
            # parent/child relationship we can disregard them for simplification
            # so here B and C aren't useful. BvC as well since A doesn't play in that match.
            # We are left with: P(A| AvB, CvA), which can be easily computed.

            # Here is how it looks like for 4 iterations of the for loop when the node to resample is A:
            # P(A = 0 | AvB, CvA)   = P(A=0) * P(AvB | A=0,B) * P(CvA | C,A=0)
            # P(A = 1 | AvB, CvA)   = P(A=1) * P(AvB | A=1,B) * P(CvA | C,A=1)
            # P(A = 2 | AvB, CvA)  = P(A=2) * P(AvB | A=2,B) * P(CvA | C,A=2)
            # P(A = 3 | AvB, CvA)   = P(A=3) * P(AvB | A=3,B) * P(CvA | C,A=3)

            # Once we get the joint probability by multiplying all the cpds,
            # WE then add the joint probability to the conditional probability list

            # I am using the same logic for B and C as well, we just have to use the respective variables
            # based of the above heuristics I used.

            if node_to_resample == "A":
                prob_a = cpd_a.values[skill_level]
                prob_avb = cpd_avb.values[initial_state[3], skill_level, initial_state[1]]
                prob_cva = cpd_cva.values[initial_state[5], initial_state[2], skill_level]
                joint_prob = prob_a * prob_avb * prob_cva
                conditional_prob.append(joint_prob)
            if node_to_resample == "B":
                prob_b = cpd_b.values[skill_level]
                prob_avb = cpd_avb.values[initial_state[3], initial_state[0], skill_level]
                prob_bvc = cpd_bvc.values[initial_state[4], skill_level, initial_state[2]]
                joint_prob = prob_b * prob_avb * prob_bvc
                conditional_prob.append(joint_prob)
            if node_to_resample == "C":
                prob_c = cpd_c.values[skill_level]
                prob_bvc = cpd_bvc.values[initial_state[4], initial_state[1], skill_level]
                prob_cva = cpd_cva.values[initial_state[5], skill_level, initial_state[0]]
                joint_prob = prob_c * prob_bvc * prob_cva
                conditional_prob.append(joint_prob)
    # if the node to resample is BvC, then we do the following:
    elif node_to_resample == "BvC":
        # here we are looping only 3 times since there are only 3 outcomes in a game
        for outcome in range(3):
            # this one is straight forward we are just getting the CPD of BvC
            # and adding to the conditional probability list
            prob_bvc = cpd_bvc.values[outcome, initial_state[1], initial_state[2]]
            conditional_prob.append(prob_bvc)

    # Normalizing the conditional probability
    # We are dividing each probability in conditional prob list by the sum of it, so their total equals 1.
    # using an array here allows for me to perform the division
    # we can divide an array by a scalar unit in the sum of conditional probability list
    # this way we can get the division of each element in the list by the sum of total list.
    conditional_prob = np.array(conditional_prob) / sum(conditional_prob)

    # we are sampling a new skill level based of the conditional probability
    new_skill_level = np.random.choice(range(len(conditional_prob)), p=conditional_prob)

    # we are updating the state with the new sampled value from above.
    updated_state = list(initial_state)
    updated_state[nodes.index(node_to_resample)] = new_skill_level

    sample = tuple(updated_state)

    return sample


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    # A_cpd = bayes_net.get_cpds("A")
    # AvB_cpd = bayes_net.get_cpds("AvB")
    # match_table = AvB_cpd.values
    # team_table = A_cpd.values
    # sample = tuple(initial_state)

    # sample a new state given the current state, unlike gibbs we can change more than one variable at a time
    new_state = list(initial_state)
    for i in range(len(initial_state)):
        # If i is 0,1,2, it will be the teams A,B,C we are representing here
        # randomly selecting 4 states for the teams A,B,C since they have 4 skill levels
        if i < 3:
            new_state[i] = random.randint(0,3)
        # if I is 3 or 5, Avb and CvA we don't do anything since they are already given to be 0 and 2.
        elif i == 3 or i == 5:
            continue
        # if I is 4, then we select BvC, we assign 3 states to it since there are 3 match outcomes.
        else:
            new_state[i] = random.randint(0,2)

    # this is where we do the acceptance probability as part of the MH sampling
    # we are calling on the helper function joint_probability which performs of new state over the current state.
    # the helper method returns the joint probability given the bayesian network and the state
    # we are dividing the joint probability of new state over initial state.

    acceptance_prob = min(1, joint_probability(bayes_net, new_state) / joint_probability(bayes_net, initial_state))

    # I created the variable sample making deep copy of initial state as a tuple
    # I did same for new_sample but instead making copy of new state as a tuple
    sample = tuple(initial_state)
    new_sample = tuple(new_state)

    # this is the rejection step, random.random() generates a random number between 0.0 and 1.0
    # if the acceptance probability is higher than random number we accept the new state as the new current state
    # we return that.
    if random.random() < acceptance_prob:
        return new_sample
    # if not then we stick to the current initial state given. I.e. the old state. return that.
    else:
        return sample

def joint_probability(bayes_net, initial_state):

    # the ratio of π(x′)/π(x) is nonetheless the " full joint probabilities, i.e., products of conditional probabilities in the Bayes net"
    # hence this is the reason for this helper method as it organizes all the products of conditional probabilities in here.

    cpd_a = bayes_net.get_cpds("A")
    cpd_b = bayes_net.get_cpds("B")
    cpd_c = bayes_net.get_cpds("C")
    cpd_avb = bayes_net.get_cpds("AvB")
    cpd_bvc = bayes_net.get_cpds("BvC")
    cpd_cva = bayes_net.get_cpds("CvA")

    # here we're setting initial probability to 1 and mulitiplying to it at each CPD. Then we return the probability.
    prob = 1
    prob *= cpd_a.values[initial_state[0]]
    prob *= cpd_b.values[initial_state[1]]
    prob *= cpd_c.values[initial_state[2]]
    prob *= cpd_avb.values[initial_state[3], initial_state[0], initial_state[1]]
    prob *= cpd_bvc.values[initial_state[4], initial_state[1], initial_state[2]]
    prob *= cpd_cva.values[initial_state[5], initial_state[2], initial_state[0]]

    return prob


def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH

    n = 1000
    delta = 0.00001

    # this is the variables for convergence checking of each sampling method, gibbs and mh.
    # they will store the last n iterations of each algorithm for checking convergence.
    convg_gibbs = []
    convg_mh = []

    # we store the BvC outcomes for both gibbs and Mh respectively in these lists.
    gibbs_convg = []
    mh_convg = []


    # the structure is the same for both as you mentioned in the readme
    # we run both gibbs and MH for n =10 times with a delta of 0.001. We do this until they converge into a
    # stationary distribution over the posterior.

    while True:
        # we are sampling gibbs, a new state is generated using the gibbs sampling function
        initial_state = Gibbs_sampler(bayes_net, initial_state)
        #we are store the BvC outcome here for gibbs
        # Gibbs_convergence[initial_state[4]] += 1
        gibbs_convg.append(initial_state[4])
        # This counter incremented everytime we sample gibbs
        Gibbs_count += 1

        # checking convergence for gibbs
        # if length of the convg_gibbs list adds upto n which is 1000, then we remove the oldest state in it.
        if len(convg_gibbs) == n:
            convg_gibbs.pop(0)
        # we then add the initial state at index 4, which is BvC value into the convergence variables.
        convg_gibbs.append(initial_state[4])

        # if the difference between the maximum and minimum of the last n iterations is less than delta, we break the loop
        # this is because if that condition is true we have reach convergence.
        if len(convg_gibbs) == n and max(set(convg_gibbs), key=convg_gibbs.count) - min(set(convg_gibbs), key=convg_gibbs.count) < delta:
            break

    # we will reset the initial state for the next part of this function, which is for MH sampling.
    initial_state = [random.randint(0,3), random.randint(0,3), random.randint(0,3), 0, random.randint(0,2), 2]

    # the MH sampling follows the exact same code structure. I will only highlight the differences.
    while True:
        # we are sampling MH
        new_state = MH_sampler(bayes_net, initial_state)
        # The MH counter also increments by 1, as we sample MH
        MH_count += 1
        # if the new state isn't same as old state, then we store the BvC outcome here for MH
        # initial state is also set to new state.
        if new_state != initial_state:
            # MH_convergence[initial_state[4]] += 1
            mh_convg.append(new_state[4])
            initial_state = new_state
        # if they are the same, then the MH rejection count variable increments by 1
        else:
            MH_rejection_count += 1
            mh_convg.append(initial_state[4])

        # checking convergence for MH
        if len(convg_mh) == n:
            convg_mh.pop(0)
        convg_mh.append(initial_state[4])

        # same logic as gibbs while loop here
        if len(convg_mh) == n and max(set(convg_mh), key=convg_mh.count) - min(set(convg_mh), key=convg_mh.count) < delta:
            break

    # Calculate convergence proportions for Gibbs
    for i in range(3):
        count = gibbs_convg.count(i)
        proportion = count / len(gibbs_convg)
        Gibbs_convergence[i] = proportion
    # Calculate convergence proportions for MH
    for i in range(3):
        count = mh_convg.count(i)
        proportion = count / len(mh_convg)
        MH_convergence[i] = proportion

    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count


def sampling_question():
    """Question about sampling performance."""
    choice = 0
    options = ['Gibbs','Metropolis-Hastings']
    factor = 0

    # initialize the bayesian network and initial state with empty list
    bayes_net = get_game_network()
    initial_state = []

    # I am reusing the randomly chosing values for my initial state from my gibbs sampling algo.
    for i in range(3):
        skill_level = random.randint(0, 3)
        initial_state.append(skill_level)

    # avb outcome
    initial_state.append(0)

    bvc_outcome = random.randint(0, 2)
    initial_state.append(bvc_outcome)

    # cva outcome
    initial_state.append(2)

    # we are trying to get the convergence information for both of sampling algos.
    Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count = compare_sampling(bayes_net, initial_state)

    # if Gibbs count is less than MH count that means, Gibbs converges faster. Hence it is our choice of what's better
    # which is why I set choice to 0, since it's in the options list, Gibbs is first index.
    # I also get the factor by just taking the ratio of Mh count over Gibbs count, it's to see how much more faster gibbs is.
    if Gibbs_count < MH_count:
        choice = 0
        factor = MH_count / Gibbs_count
    # if mh count is less than gibbs count, then MH converges fasters
    # hence why I made it right choice and factor is similar to above but with flipped variables.
    elif MH_count < Gibbs_count:
        choice = 1
        factor = Gibbs_count / MH_count
    # if both converge at the same rate, then we set factor to 1, as they took same number of iterations to converge.
    else:
        factor = 1

    return options[choice], factor
