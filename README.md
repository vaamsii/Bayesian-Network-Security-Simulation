
# Bayesian Network Project: MI6 Security System Simulation

## Overview
In this project, I explore probabilistic models using Bayesian networks to efficiently calculate probabilities concerning discrete random variables. The focus is on a hypothetical scenario involving MI6's security system where a criminal organization attempts to access classified information.

## Theoretical Background
The project leverages fundamental concepts from the fields of "Quantifying Uncertainty" and "Probabilistic Reasoning." **Techniques such as Markov Chain Monte Carlo, Gibbs Sampling, and Metropolis Hastings Sampling are integral to the simulation processes.** These methods are crucial for understanding the dynamics and probabilities of complex systems like the one modeled here.

## Project Components

### Bayesian Network Design
I designed a Bayesian network that models the espionage activities by the criminal organization aimed at breaching MI6's security. The network includes nodes representing key actions and states that lead up to the possible compromise of secret files. Here's the breakdown of the node attributes used in the Bayes Network:

- **H**: Event where the organization hires hackers
- **C**: Event where the organization acquires a computer system called "Contra"
- **M**: Event where the organization hires mercenaries
- **B**: Event where Bond is guarding M
- **Q**: Event where Q’s database is hacked
- **K**: Event where M is kidnapped
- **D**: Event where the organization accesses the “Double-0” files

### Setting Probabilities
Conditional probabilities for the network were set using the `pgmpy` package. This involved defining the likelihoods of each event based on historical data and hypothetical scenarios. Here are the tables defining some of these probabilities:

**Probability of the organization not finding skilled hackers (H = false)**:
- True: 0.5

**Probability of the organization acquiring Contra (C = true)**:
- True: 0.3

**Probability of unsuccessful hiring of mercenaries (M = false)**:
- True: 0.2

**Probability of Bond guarding M at any time (B = true)**:
- True: 0.5

These probabilities guide the simulations and help predict the outcome of different scenarios.

### Probability Calculations
The main tasks involve:
1. Calculating the marginal probability that the “Double-0” files get compromised.
2. Evaluating how updates, like securing critical systems or changes in personnel assignments, affect these probabilities.

### Simulation and Sampling Techniques
I employed two main sampling techniques to estimate the outcome probabilities:

- **Gibbs Sampling**: This method was used to simulate the network's probability distribution, focusing on reaching a stationary distribution after several iterations.
  
- **Metropolis-Hastings Sampling**: This sampling technique builds an approximation of the latent probability distribution by generating and evaluating multiple candidate states.

These techniques are crucial for understanding the probability distributions in scenarios that are computationally expensive or impractical to calculate directly.

## Conclusion
This project demonstrates the application of Bayesian networks in simulating complex security scenarios. Through probabilistic modeling and advanced sampling methods, I provide insights into how different factors influence the security of sensitive information in a high-stakes environment. The project not only enhances my understanding of Bayesian networks but also showcases the practical implications of probabilistic reasoning in real-world scenarios.
