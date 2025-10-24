---
layout: post
title: "Introduction to Markov Chains"
category: theory
tags: maths probability
mermaid: true
katex: true
---

<blockquote cite="Mark V. Shaney">
I hope that there are sour apples in every bushel.
</blockquote>

TL;DR
- Scope: homogeneous discrete first-order Markov chains
- Keywords: Chapman-Kolmogorov, hitting times, invariant distribution, random walk
- References: This is essentially an abridged version of [Dexter's notes](https://dec41.user.srcf.net/notes/IB_M/markov_chains.pdf) of [Grimmett's](https://www.statslab.cam.ac.uk/~grg/) lectures

## Definition

A Markov chain is a sequence of random variables \\(X = (X_0, X_1, \dots)\\) taking values in a discrete state space \\(S\\) such that \\(\mathbb{P}(X_0 = S_i) = \lambda_i\\) and

$$\mathbb{P}(X_{n+1} = x_{n+1} \mid X_0 = x_0, \dots, X_n = x_n) = \mathbb{P}(X_{n+1} = x_{n+1} \mid X_n = x_n)$$

Intuitively, if we want to know the probability of transitioning to a new state, it suffices to know only the current state - the rest of the history is irrelevant.

For example, if we flip a coin and score \\(+1\\) for heads and \\(-1\\) for tails, then the accumulated scores after each flip form a Markov chain. Let's make the state space finite by stopping when the score hits \\(\pm 3\\), such that \\(S=\\{-3,-2,-1,0,+1,+2,+3\\}\\). Then we can construct the following graph:

<div class="mermaid" style="text-align: center;">
graph LR
    Start("0")
    Plus1(("+1"))
    Plus2(("+2"))
    Plus3(("+3"))
    Minus1(("-1"))
    Minus2(("-2"))
    Minus3(("-3"))
    
    Start -->|0.5| Plus1
    Start -->|0.5| Minus1
    
    Plus1 -->|0.5| Plus2
    Plus1 -->|0.5| Start
    
    Plus2 -->|0.5| Plus3
    Plus2 -->|0.5| Plus1
    
    Minus1 -->|0.5| Start
    Minus1 -->|0.5| Minus2
    
    Minus2 -->|0.5| Minus1
    Minus2 -->|0.5| Minus3
    
    Plus3 -->|1| Plus3
    Minus3 -->|1| Minus3
    
    style Plus3 fill:#90EE90
    style Minus3 fill:#FFB6C6
    style Start fill:#87CEEB
</div>

This is equivalent to the transition matrix \\(P\\), where \\(p_{ij} = \mathbb{P}(X_{n+1} = S_j \mid X_n = S_i)\\).

$$
P = \begin{pmatrix}
\colorbox{#FFB6C6} 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0.5 & 0 & 0.5 & 0 & 0 & 0 & 0 \\
0 & 0.5 & 0 & 0.5 & 0 & 0 & 0 \\
0 & 0 & 0.5 & \colorbox{#87CEEB} 0 & 0.5 & 0 & 0 \\
0 & 0 & 0 & 0.5 & 0 & 0.5 & 0 \\
0 & 0 & 0 & 0 & 0.5 & 0 & 0.5 \\
0 & 0 & 0 & 0 & 0 & 0 & \colorbox{#90EE90} 1
\end{pmatrix}
$$

Each row sums to 1, defining a discrete distribution from the corresponding state, and the states \\(\pm 3\\) are absorbing: once we hit them, we never leave.

## n-step transitions

Suppose we know the current state and want to figure out the distribution over \\(S\\) after not one, but two, coin flips. To get from \\(x_0\\) to \\(x_2\\), it suffices to marginalise over all possible paths \\(x_0 \rightarrow x_1 \rightarrow x_2\\) for each \\(x_1 \in S\\).

$$
\begin{aligned}
\mathbb{P}(X_2 = x_2 \mid X_0 = x_0) &= \sum_{x_1 \in S} \mathbb{P}(X_2 = x_2, X_1 = x_1 \mid X_0 = x_0) \\
&= \sum_{x_1 \in S} \mathbb{P}(X_2 = x_2 \mid X_1 = x_1, X_0 = x_0) \mathbb{P}(X_1 = x_1 \mid X_0 = x_0) \\
&= \sum_{x_1 \in S} \mathbb{P}(X_2 = x_2 \mid X_1 = x_1) \mathbb{P}(X_1 = x_1 \mid X_0 = x_0) \\
&= \sum_{0 \leq k < |S|} p_{kj} p_{ik}
\end{aligned}
$$

where \\(x_0 = S_i, x_2 = S_j\\), and the third line utilises the Markov property. This is just a matrix multiplication, and so the two-step transition matrix \\(P(2)\\) is simply:

$$
P^2 = \begin{pmatrix}
\colorbox{#FFB6C6} 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0.5 & 0.25 & 0 & 0.25 & 0 & 0 & 0 \\
0.25 & 0 & 0.5 & 0 & 0.5 & 0 & 0 \\
0 & 0.25 & 0 & \colorbox{#87CEEB} {0.5} & 0 & 0.25 & 0 \\
0 & 0 & 0.25 & 0 & 0.5 & 0 & 0.25 \\
0 & 0 & 0 & 0.25 & 0 & 0.25 & 0.5 \\
0 & 0 & 0 & 0 & 0 & 0 & \colorbox{#90EE90} 1
\end{pmatrix}
$$

If you're fancy-pants, you can generalise this to get the discrete Chapman-Kolmogorov equation \\(P(m+n) = P(m)P(n)\\), but in our (homogeneous) case we're happy with \\(P(n) = P^n\\).

What happens as \\(n \rightarrow \infty\\)? Let's consider a simpler transition matrix

$$
P = \begin{pmatrix}
1-\alpha & \alpha\\
\beta & 1-\beta\\
\end{pmatrix}
$$ 

In order to calculate \\(P^n\\), we first diagonalise it. We solve

$$
\begin{aligned}
0 &= \det(P - \lambda I)\\
&= (1 - \alpha - \lambda)(1 - \beta - \lambda) - \alpha\beta\\
&= \lambda^2 - (2 - \alpha - \beta)\lambda + (1 - \alpha - \beta)\\
&= (\lambda - 1)(\lambda - (1 - \alpha - \beta))
\end{aligned}
$$

to get eigenvalues \\(\lambda_0 = 1, \lambda_1 = 1 - \alpha - \beta\\), such that

$$
P^n = U^{-1}
\begin{pmatrix}
\lambda_0^n & 0\\
0 & \lambda_1^n\\
\end{pmatrix}
U
$$

for some matrix \\(U\\). We can skip calculating \\(U\\) and go straight to the linear form

$$
p_{1,2}(n) = A\cancel{\lambda_0^n} + B\lambda_1^n
$$

Plugging in \\(p_{1,2}(0) = 0\\) and \\(p_{1,2}(1) = \alpha\\), we get

$$
\begin{aligned}
0 &= A + B\\
\alpha &= A + B\lambda_1
\end{aligned}
$$

which gives

$$
p_{1,2}(n) = \frac{\alpha}{\alpha + \beta} (1 - \lambda_1^n)
$$

This is all the algebra we need! We swap \\(\alpha \leftrightarrow \beta\\) to get \\(p_{2,1}\\) and remember that rows sum to 1 to get

$$
P^n = \frac{1}{\alpha + \beta}
\begin{pmatrix}
\beta + \alpha\lambda_1^n & \alpha(1 - \lambda_1^n)\\
\beta(1 - \lambda_1^n) & \alpha + \beta\lambda_1^n\\
\end{pmatrix}
\rightarrow
\frac{1}{\alpha + \beta}
\begin{pmatrix}
\beta & \alpha\\
\beta & \alpha\\
\end{pmatrix}
$$

as \\(n \rightarrow \infty\\). All the rows are equal, which means that all starting points converge to a common distribution over \\(S\\)! We call this the invariant distribution \\(\pi_i = p_{ji}(\infty)\\) for all \\(j\\). Note that \\(\pi P = \pi\\), as expected.

## Hitting statistics

Given a Markov chain \\((X_n)_{n \geq 0}\\) and a subset \\(A \subseteq S\\) of the state space, we define the hitting time \\(H^A = \min(\\{n \geq 0: X_n \in A\\} \cup \\{\infty\\})\\) and the hitting probability \\(h_i^A = \mathbb{P}(H^A < \infty \mid X_0 = S_i)\\).

**Theorem 1.** The vector \\(h^A = (h_i^A: S_i \in S)\\) satisfies

$$
h_i^A =
\begin{cases}
1 &\text{if } S_i \in A\\
\sum_{0 \leq j < |S|} p_{ij} h_j^A &\text{if } S_i \notin A
\end{cases}
$$

and is minimal, in that for any non-negative solution \\((x_i^A: S_i \in S)\\) of these equations, we have \\(h_i^A \leq x_i\\) for all \\(i\\).

**Proof.** Read [Dexter's notes](https://dec41.user.srcf.net/notes/IB_M/markov_chains.pdf).

We get a similar result for the expected hitting time \\(k_i = \mathbb{E}[H^A \mid X_0 = S_i]\\).

**Theorem 2.** The vector \\(k^A = (k_i^A: S_i \in S)\\) is the minimal solution to

$$
k_i^A =
\begin{cases}
0 &\text{if } S_i \in A\\
1 + \sum_{0 \leq j < |S|} p_{ij} k_j^A &\text{if } S_i \notin A
\end{cases}
$$

**Proof.** Similar to before, except that we add \\(1\\) for the step \\(i \rightarrow j\\).

This often manifests itself in the form of a set of difference equations. Let's illustrate this by returning to the coin flip example, with \\(A=\\{-3\\}\\), i.e. what is the chance of 'losing'? Trivially, \\(h_0 = 1, h_6 = 0\\). Following Theorem 1, we must satisfy

$$h_i = \frac{1}{2}(h_{i-1} + h_{i+1})$$

for all \\(0 < i < 6\\). This is a second-order difference equation:

$$0 = h_{i-1} - 2h_i + h_{i+1}$$

Plugging in \\(h_i = r^i\\) yields the characteristic equation

$$0 = \cancel{r^{i-1}}(r-1)^2$$

with repeated roots at \\(r=1\\), so \\(h_i\\) is linear in \\(i\\). Plugging in the boundary conditions, this resolves to

$$h_i = 1 - \frac{i}{6}$$

What happens if we remove the absorbing state at \\(S_6 = +3\\) such that \\(S = \\{-3, -2, -1\\} \cup \mathbb{N}\\)?

Without the boundary condition on the right, the only non-negative solution is \\(h_i = 1\\) for all \\(i\\)! This means that, no matter where we start from, we will lose with 100% certainty - this is known as gambler's ruin.

**Exercise.** Solve for expected hitting times \\(k_i\\).

**Exercise.** Solve for a general coin flip with probability \\(p\\) of landing heads.

## Conclusion

These are some of the more basic, canonical results derived using Markov chains. However, we've now laid a foundation for many more applications, ranging from Hidden Markov Models to sampling Bayesian posteriors, the Wiener process (Brownian motion), Google's PageRank algorithm, and n-grams in NLP (the strange quote at the top of this post was generated using a [Markov chain model](https://www.clear.rice.edu/comp200/13spring/notes/18/shaney.shtml))!

<!--Draft
- Scope: homogeneous discrete 1st-order Markov chains
- Markov property, distribution, transition matrix
- Extended Markov property [skip]
- Chapman-Kolmogorov
- Communicating classes, irreducable [skip]
- Recurrence/transience, Abel's lemma, Polya's theorem, ergodic [skip]
- Hitting prob/time
- Gambler's ruin, birth-death
- Weak/strong Markov [skip]
- Invariant distribution
- Reversible chains [skip]

Keywords
- MCMC
- Discrete/continuous
- Transition matrix
- Stationary
- Hitting time/prob
- Text generation
- Random walk, Wiener process, Brownian motion, Poisson process
- 100 pirate brainteaser?
- Ergodicity, strong Markov, recurrence/transience
- Champan-Kolmogorov
- HMM, Queueing, PageRank
- Modeling corners

Trivia
- Mark V. Shaney https://www.clear.rice.edu/comp200/13spring/notes/18/shaney.shtml -->
