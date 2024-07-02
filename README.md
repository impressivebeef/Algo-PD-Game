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

### State Space

### Updating Rules
$$
 Q_i^{t+1}(a_t|s_t) = (1-\alpha)*Q_i^t(a_t|s_t) + \alpha * (\pi_{t+1} + \delta * max_{a^\prime}(Q_i^t(a^\prime|s_{t+1})))
$$
### Policy
$$
\epsilon_t = e^{-\beta t}
$$
## Simulation code
