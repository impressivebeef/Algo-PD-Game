Hey there! I created this repository to upload the python code I wrote for my thesis. The code is in large the same as the code that I used for my thesis, however I did make some minor changes to make it run more smoothly on other systems. I might continue to make minor changes here and there from time to time.

There are 2 files in the repository: Algo PD Game Simulation.py and Results Reader.py. The main simulation file is Algo PD Game Simulation.py. When you run in it you first get asked to assign a directory for the program to store the simulation results. It can handle the standard windows format of `C:\Users\Name\Documents\etc` fine, however make sure to end the path with the desired name of the csv file. So for example: `C:\Users\Name\Documents\results.csv`. 

After the program asks which simulation you would like to run. There are 2 types: No price discrimination and price discrimination (see below what exactly the differences between them are). Next it will ask whether you want to use default or custom settings. The default setting are the values I used in my thesis. For custom you must specify the number of runs, the end period (T), the convergence criteria (M), the number of consumers (N) and the exploration paremeter ($\beta$). The program will then take these values and run the simulation. The simulation returns a df containing the following: run count, period count, convergence (True or false), convergence count, state, consumer surplus, profit firm A and profit firm B, which will then be saved as a csv file. 

The prices are not directly stored in the df in order to save memory, however they can still be accessed by using Results Reader.py. When you run Results reader.py you first must specify the location of the csv file containing the simulation results. After you must specify the location for the new csv file that Results Reader.py will generate. Lastly it will ask which simulation type it is reading (i.e. no price discrimination and price discrimination). This programs returns a csv file containing all the original columns with now price data of both firms for each period.


# Algorithimic Price Discrimination Game



## Economic Model

The economic model is inspired by the [Esteves (2014)](https://onlinelibrary-wiley-com.tilburguniversity.idm.oclc.org/doi/full/10.1111/sjoe.12061) model of price discrimination with private and imperfect information. 

Assume a Hotelling model with 2 firms located at each end point of interval $l \in[-\frac{1}{2},\frac{1}{2}]$, where $l$ represents brand loyalty. Each firm offers a good at price $P_j$ where j denotes the corresponding firm. There are $N$ consumers uniformly distributed over interval $l$. The closer a consumer is to a firm the more "loyal" the customer is to that firm, and therefore more likely to buy from that firm. Each consumer has an identical private value $v$ and incur a "loyalty" cost of $|l_i-l_j|$ between their location and the location of the firm, where i denotes the corresponding consumer and j the firm. So the consumer will buy the good if:

$$
 v-|l_i-l_j| > P_j
$$

Furthermore, the consumer will buy from the firm that maximises their own utility. Once the consumer buys a good, the consumer will gain utility equal to  their valuation minus the price and loyalty cost:

$$
 U_i = v  - P_j - |l_i-l_j|
$$

Total consumer surplus is then the total sum of the utility of all consumers:

$$
CS = \Sigma_{i=1}^N U_i  
$$

Their respective location $l_i$ is known to the consumer, but unobserved by the firms. Each firm receives a signal that can take the form of $L$ indicating a consumer is located between $[-\frac{1}{2},0]$ or $R$, a consumer is located between $[0,\frac{1}{2}]$. This signal is "noisy" and has a probability of returning a wrong signal. The probability of receiving a right signal is correlated with the true location of the consumer:

$$
p(R|l_i) = \frac{1}{2} + b*l_i 
$$

$$
p(L|l_i) = 1-p(R|l_i) =\frac{1}{2}-b*l_i 
$$

With $b \in [0,1]$ denoting the strength of the signal. Upon receiving the signal the firms can decide what price to charge the consumer, drawn from a discrete distribution $P \in [p_0,p_1,...,p_k]$. Both firms receive a separate signal independent from each other and otherwise do not observe which signal the competing firm has received. However, based on their own private signal they update their beliefs about the competing firms' signal. For simplicity, the firms do not face any fixed nor marginal costs.



## Q-Learning Algorithm

### Action Space

In the no PD case the action set $A_{noPD}$ contains all potential prices that the firm can set from price distribution $P$. In the PD case, the action set $A_{PD}$ consists of all possible bundles of prices $(P_L,P_R)$ where $P_L$ and $P_R$ denotes the prices for signal $L$ and $R$ respectively. The algorithm decides on which action $a_j \in A_{PD}$ to undertake  before observing the signal.

### State Space

Each state is a combination of the actions $a$ undertaken by both algorithms. In the no PD case the algorithm observes its own and the competitors price as the state. As for the PD case, the algorithm observes the price bundles set by itself and the competitor. The state space has a one-period memory, hence every period the previous state gets replaced with the new state. A one-period memory is implemented to restrict the cardinality of the state space. 

### Updating Rules

The algorithm update the expected value of the corresponding state-action cell using the following equation:

$$
 Q_i^{t+1}(a_t|s_t) = (1-\alpha)*Q_i^t(a_t|s_t) + \alpha * (\pi_{t+1} + \delta * max_{a^\prime}(Q_i^t(a^\prime|s_{t+1})))
$$
### Policy

The algorithm determines its action based on whether it's exploring or exploiting. During exploration, it selects a random action $a_j \in A$. During exploitation, it selects the action that maximizes the expected return based on the current state. To put it formally, during exploitation, the algorithm selects the action $a_j$ according to: $max_a(Q^t_i(a|s_t))$. The decision between exploring and exploiting is dictated by the policy of the algorithm. This project adopts the time-declining $\epsilon$-greedy policy as described in [Calvano et al. (2020)](https://www-aeaweb-org.tilburguniversity.idm.oclc.org/articles?id=10.1257/aer.20190623):

$$
\epsilon_t = e^{-\beta t}
$$

 Where $\beta > 0$ is a parameter that dictates the rate of exploration decline. $\epsilon$ determines whether the algorithm will explore or exploit. There is an $\epsilon$ chance that the algorithm will explore and a $(1-\epsilon)$ that the algorithm will exploit. A good rule of thumb for setting $\beta$ is to set it such that at the end period $T$ the chance for exploring is 1 over the convergence criteria I.E. $\epsilon_{t=T} = \frac{1}{M}$. This specification ensures runs only convergence in the end stages of a run, allowing for higher levels of learning.

## Simulation code

The simulation code in Algo PD Game Simulation.py uses the following for-loop structure for both the no PD and PD case:

| Pseudo Code                                                      |
|------------------------------------------------------------------|
| 1 Initialise Q-Matrices QA, QB to Q0                             |
| 2 Set state S0 randomly                                          |
| 3 Initialise t = 0, m = 0                                        |
| **3 Start the simulation run**                                   |
| 4 \| Determine actions aA, aB                                    |
| 5 \| Calculate payouts πA, πB                                    |
| 6 \| Determine state St+1                                        |
| 7 \| Update state-action cells QA(aA\|St), QA(aB\|St)            |
| 8 \| Check if states St = St+1, if yes: convergence count m = +1 |
| 9 \| Add period count t = +1                                     |
| **10 Check if t = T or m = M , if not: loop back to step 3**     |
