import unittest
from bayes_network import *
"""
Contains various local tests for Assignment 3.
"""

class ProbabilityTests(unittest.TestCase):

    #Part 1a͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
    def test_network_setup(self):
        """Test that the power plant network has the proper number of nodes and edges."""
        security_system = make_security_system_net()
        nodes = security_system.nodes()
        self.assertEqual(len(nodes), 7, msg="incorrect number of nodes")
        total_links = security_system.number_of_edges()
        self.assertEqual(total_links, 6, msg="incorrect number of edges between nodes")

    #Part 1b͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
    def test_probability_setup(self):
        """Test that all nodes in the power plant network have proper probability distributions.
        Note that all nodes have to be named predictably for tests to run correctly."""
        # test H distribution͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
        security_system = set_probability(make_security_system_net())
        H_node = security_system.get_cpds('H')
        self.assertTrue(H_node is not None, msg='No H node initialized')

        H_dist = H_node.get_values()
        self.assertEqual(len(H_dist), 2, msg='Incorrect H distribution size')
        test_prob = H_dist[0]
        self.assertEqual(round(float(test_prob*100)), 50, msg='Incorrect H distribution')

        # test C distribution͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
        security_system = set_probability(make_security_system_net())
        C_node = security_system.get_cpds('C')
        self.assertTrue(C_node is not None, msg='No C node initialized')

        C_dist = C_node.get_values()
        self.assertEqual(len(C_dist), 2, msg='Incorrect C distribution size')
        test_prob = C_dist[0]
        self.assertEqual(round(float(test_prob*100)), 70, msg='Incorrect C distribution')

        # test M distribution͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
        security_system = set_probability(make_security_system_net())
        M_node = security_system.get_cpds('M')
        self.assertTrue(M_node is not None, msg='No M node initialized')

        M_dist = M_node.get_values()
        self.assertEqual(len(M_dist), 2, msg='Incorrect M distribution size')
        test_prob = M_dist[0]
        self.assertEqual(round(float(test_prob*100)), 20, msg='Incorrect M distribution')

        # test B distribution͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
        security_system = set_probability(make_security_system_net())
        B_node = security_system.get_cpds('B')
        self.assertTrue(B_node is not None, msg='No B node initialized')

        B_dist = B_node.get_values()
        self.assertEqual(len(B_dist), 2, msg='Incorrect B distribution size')
        test_prob = B_dist[0]
        self.assertEqual(round(float(test_prob*100)), 50, msg='Incorrect B distribution')


        # Q distribution͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
        # can't test exact probabilities because͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
        # order of probabilities is not guaranteed͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
        Q_node = security_system.get_cpds('Q')
        self.assertTrue(Q_node is not None, msg='No Q node initialized')
        [cols, rows1, rows2] = Q_node.cardinality
        self.assertEqual(rows1, 2, msg='Incorrect Q distribution size')
        self.assertEqual(rows2, 2, msg='Incorrect Q distribution size')
        self.assertEqual(cols,  2, msg='Incorrect Q distribution size')

        # K distribution͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
        K_node = security_system.get_cpds('K')
        self.assertTrue(K_node is not None, msg='No K node initialized')
        [cols, rows1, rows2] = K_node.cardinality
        self.assertEqual(rows1, 2, msg='Incorrect K distribution size')
        self.assertEqual(rows2, 2, msg='Incorrect K distribution size')
        self.assertEqual(cols,  2, msg='Incorrect K distribution size')

        # D distribution͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
        D_node = security_system.get_cpds('D')
        print(D_node)
        self.assertTrue(D_node is not None, msg='No D node initialized')
        [cols, rows1, rows2] = D_node.cardinality
        self.assertEqual(rows1, 2, msg='Incorrect D distribution size')
        self.assertEqual(rows2, 2, msg='Incorrect D distribution size')
        self.assertEqual(cols,  2, msg='Incorrect D distribution size')
        try:
            security_system.check_model()
        except:
            self.assertTrue(False, msg='Sum of the probabilities for each state is not equal to 1 or CPDs associated with nodes are not consistent with their parents')


    #Part 2a Test͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
    def test_games_network(self):
        """Test that the games network has the proper number of nodes and edges."""
        games_net = get_game_network()
        # print(games_net)
        nodes = games_net.nodes()
        # print(nodes)
        self.assertEqual(len(nodes), 6, msg='Incorrect number of nodes')
        total_links = games_net.number_of_edges()
        # print(total_links)
        self.assertEqual(total_links, 6, 'Incorrect number of edges')

        # Now testing that all nodes in the games network have proper probability distributions.͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
        # Note that all nodes have to be named predictably for tests to run correctly.͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁

        # First testing team distributions.͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
        # You can check this for all teams i.e. A,B,C (by replacing the first line for 'B','C')͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁

        A_node = games_net.get_cpds('A')
        print(A_node)
        self.assertTrue(A_node is not None, 'Team A node not initialized')
        A_dist = A_node.get_values()
        print(A_dist)
        self.assertEqual(len(A_dist), 4, msg='Incorrect distribution size for Team A')
        test_prob = A_dist[0]
        test_prob2 = A_dist[2]
        self.assertEqual(round(float(test_prob*100)),  15, msg='Incorrect distribution for Team A')
        self.assertEqual(round(float(test_prob2*100)), 30, msg='Incorrect distribution for Team A')

        # Now testing match distributions.͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
        # You can check this for all matches i.e. AvB,BvC,CvA (by replacing the first line)͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
        AvB_node = games_net.get_cpds('AvB')
        print(AvB_node)
        BvC_node = games_net.get_cpds('BvC')
        print(BvC_node)
        CvA_node = games_net.get_cpds('CvA')
        print(CvA_node)
        self.assertTrue(AvB_node is not None, 'AvB node not initialized')

        AvB_dist = AvB_node.get_values()
        print(AvB_dist)
        BvC_dist = BvC_node.get_values()
        print(BvC_dist)
        CvA_dist = CvA_node.get_values()
        print(CvA_dist)
        [cols, rows1, rows2] = AvB_node.cardinality
        self.assertEqual(rows1, 4, msg='Incorrect match distribution size')
        self.assertEqual(rows2, 4, msg='Incorrect match distribution size')
        self.assertEqual(cols,  3, msg='Incorrect match distribution size')

        flag1 = True
        flag2 = True
        flag3 = True
        for i in range(0, 4):
            for j in range(0,4):
                x = AvB_dist[:,(i*4)+j]
                if i==j:
                    if x[0]!=x[1]:
                        flag1=False
                if j>i:
                    if not(x[1]>x[0] and x[1]>x[2]):
                        flag2=False
                if j<i:
                    if not (x[0]>x[1] and x[0]>x[2]):
                        flag3=False

        self.assertTrue(flag1, msg='Incorrect match distribution for equal skill levels')
        self.assertTrue(flag2 and flag3, msg='Incorrect match distribution: teams with higher skill levels should have higher win probabilities')

    #Part 2b Test͏󠄂͏️͏󠄌͏󠄎͏︈͏͏󠄁
    def test_posterior(self):
        posterior = calculate_posterior(get_game_network())
        print(posterior)

        self.assertTrue(abs(posterior[0]-0.25)<0.01 and abs(posterior[1]-0.42)<0.01 and abs(posterior[2]-0.31)<0.01, msg='Incorrect posterior calculated')

    def test_gibbs_sampler_convergence(self):
        bayes_net = get_game_network()
        iterations = 1000000
        counts = {0: 0, 1: 0, 2: 0}

        # Initial state - AvB and CvA are fixed, others can be random
        # sample = [random.randint(0, 3), random.randint(0, 3), random.randint(0, 3), 0, random.randint(0, 2), 2]
        sample = [0,0,0,0,0,2]

        # print(sample)
        # print(bayes_net)

        for _ in range(iterations):
            sample = Gibbs_sampler(bayes_net, sample)
            counts[sample[4]] += 1  # Update count for BvC outcome

        # Convert counts to proportions
        proportions = {k: v / iterations for k, v in counts.items()}
        print(proportions)

        # Assertions for checking if proportions are close to expected values from 2b
        self.assertTrue(abs(proportions[0] - 0.25) < 0.01, "Proportion for 0 is not within expected range")
        self.assertTrue(abs(proportions[1] - 0.42) < 0.01, "Proportion for 1 is not within expected range")
        self.assertTrue(abs(proportions[2] - 0.31) < 0.01, "Proportion for 2 is not within expected range")

        # Print proportions for manual verification
        print("Proportions after Gibbs sampling:", proportions)

    def test_mh_sampler_convergence(self):
        bayes_net = get_game_network()
        iterations = 1000000
        counts = {0: 0, 1: 0, 2: 0}

        # Initial state - AvB and CvA are fixed, others can be random
        # It's important to start from a valid state
        sample = [random.randint(0, 3), random.randint(0, 3), random.randint(0, 3), 0, random.randint(0, 2), 2]

        for _ in range(iterations):
            sample = MH_sampler(bayes_net, sample)
            counts[sample[4]] += 1  # Update count for BvC outcome

        # Convert counts to proportions
        proportions = {k: v / iterations for k, v in counts.items()}
        print(proportions)

        # Assertions for checking if proportions are close to expected values from 2b
        self.assertTrue(abs(proportions[0] - 0.25) < 0.01, "Proportion for 0 is not within expected range")
        self.assertTrue(abs(proportions[1] - 0.42) < 0.01, "Proportion for 1 is not within expected range")
        self.assertTrue(abs(proportions[2] - 0.31) < 0.01, "Proportion for 2 is not within expected range")

        # Print proportions for manual verification
        print("Proportions after Metropolis-Hastings sampling:", proportions)

    def test_compare_sampling_convergence(self):
        bayes_net = get_game_network()
        initial_state = [random.randint(0, 3) for _ in range(3)] + [0, random.randint(0, 2), 2]

        # Perform comparison of Gibbs and MH samplings
        Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count = compare_sampling(bayes_net,
                                                                                                        initial_state)
        print(compare_sampling(bayes_net, initial_state))

        # choice, factor = sampling_question()
        #
        # print(choice)
        # print(factor)
        # print(sampling_question())

        # Check if the Gibbs and MH counts are within a reasonable range
        self.assertTrue(Gibbs_count > 0, "Gibbs sampler did not iterate enough times")
        self.assertTrue(MH_count > 0, "MH sampler did not iterate enough times")

        # Check if the samplers have converged
        self.assertTrue(len(Gibbs_convergence) == 3, "Gibbs sampler did not converge correctly")
        self.assertTrue(len(MH_convergence) == 3, "MH sampler did not converge correctly")

        # Optional: Check the rejection rate of MH sampler
        self.assertTrue(MH_rejection_count < MH_count, "MH sampler rejection rate is too high")

    def test_sampling_question(self):

        # Now call the sampling_question function
        choice, factor = sampling_question()

        print(choice)
        print(factor)
        print(sampling_question())

        # Check if the choice and factor are set appropriately
        # You may need to adjust this test depending on what you expect sampling_question to return.
        # For example, if Gibbs converges faster:
        if choice == "1":
            self.assertEqual(choice, 'Gibbs', "Incorrect choice")
        self.assertGreater(factor, 0, "Factor should be positive")

        # If Metropolis-Hastings converges faster:
        if choice == "0":
            self.assertEqual(choice, 'Metropolis-Hastings', "Incorrect choice")
        self.assertGreater(factor, 0, "Factor should be positive")
    def test_get_marginal_double0(self):
        """Test the marginal probability of Double-0 files being compromised."""
        bayes_net = set_probability(make_security_system_net())
        prob = get_marginal_double0(bayes_net)
        print(prob)
        # Assuming you know the expected range or value
        # self.assertTrue(0 <= prob <= 1, msg="Probability is not within the valid range")
    def test_get_conditional_double0_given_no_contra(self):
        """Test the conditional probability of Double-0 files being compromised with Contra shut down."""
        bayes_net = set_probability(make_security_system_net())
        prob = get_conditional_double0_given_no_contra(bayes_net)
        print(prob)
        # Assuming you know the expected range or value
        self.assertTrue(0 <= prob <= 1, msg="Probability is not within the valid range")

    def test_get_conditional_double0_given_no_contra_and_bond_guarding(self):
        """Test the conditional probability of Double-0 files being compromised with Contra shut down and Bond guarding M."""
        bayes_net = set_probability(make_security_system_net())
        prob = get_conditional_double0_given_no_contra_and_bond_guarding(bayes_net)
        print(prob)
        # Assuming you know the expected range or value
        self.assertTrue(0 <= prob <= 1, msg="Probability is not within the valid range")


if __name__ == '__main__':
    unittest.main()
