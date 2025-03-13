---
layout: post
title:  "[1] A deep dive into OthelloGPT: Sprint"
date:   2025-03-05 22:13:00 +0000
categories: jekyll update
---
<script type="text/javascript" id="MathJax-script" async
    src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
<link rel="stylesheet" href="/assets/css/style.css">

# Plan
Should the style be chronological investigation or executive summary? I reckon chronological is nice as it's different from the summary.
- Failed experiments
  - Questions I wanted to answer
  - Different model sizes
  - Loadsa targets: conditional and inductive, flips
- PE probe
  - PTEM to PE
  - PE-E vs "just played" target
  - canonical basis: t-SNE/colinearity
- L2H5 empty head
- Conclusion
- Reflections

This post covers the content of a research sprint I did last month on OthelloGPT and mechanistic interpretability, following an initial [Replication]({% post_url 2025-03-02-othello-gpt-0 %}) project. I show that prior work on linear probes can be extended by discovering new probes and using them to interpret an attention head that's part of the mechanism that constructs OthelloGPT's world model! Aside from some preliminary work on research directions that I decided not to pursue for the sprint, this work was done over 16 hours in 2 days.

# Table of Contents
<div style="border: 1px solid #ccc; background-color:rgb(239, 251, 255); padding: 10px; margin-bottom: 10px; width: 250px">
  <ul>
    <li><a href="#preliminary-work">Preliminary work</a>
      <ul>
        <li><a href="#status-quo">Status quo</a></li>
        <li><a href="#inductive-probes">Inductive probes</a></li>
        <li><a href="#conditional-probes">Conditional probes</a></li>
      </ul>
    </li>
    <li><a href="#sprint">Sprint</a></li>
    <li><a href="#conclusion">Conclusion</a></li>
  </ul>
</div>

# Preliminary work

#### Status quo

I picked up from the previous post replicating Kenneth Li and Neel Nanda with:

  - A 6M param GPT2-style model (8 layers, 8 heads, 256 dimensions) that could predict legal next moves in 6x6 Othello games 99.97% of the time
  - A linear probe (EE) that could predict whether a board square is empty with 99.95% accuracy using only the model's residual stream at L0_mid
  - A linear probe (T-M) that could predict whether a non-empty board square belongs to the opponent with 99.2% accuracy at L6_mid
  - A visualisation identifying a neuron that used the board state identified by the two probes above to boost the unembedding (U) logit for a corresponding legal move

This was my first attempt at doing mech interp research, so rather than answering more high level questions, such as "what makes a good feature?" or "how do we interpret superposition in transformers?", I went with a simpler approach that lent itself more towards playing around with the model. I set out to answer the question: **"how does OthelloGPT compute its world model?"**

I had a hunch that the model was using additional latent features in order to compute the board state. I had some weak evidence for this:

  - At the end of my previous post, I discovered that superposition between the board state probes (EE & T-M) and the unembedding vectors (U) was a likely cause for reduced probe accuracy across the model's final few layers.
<img src="/assets/images/othello-gpt/l5n415_in.png" height="120px"/>
<img src="/assets/images/othello-gpt/l5n415_out.png" height="120px"/>

  - The repertoire of feature vectors that had been discovered amounted to at most 163 dimensions out of 256 available.

| Feature vector          | Count |
|-------------------------|-------|
| Token embed (B)         | 32    |
| Pos embed (P)           | 31    |
| Empty probe (EE)        | 32    |
| Theirs/mine probe (T-M) | 36    |
| Token unembed (U)       | 32    |
|-------------------------|-------|
| **Total**               | 163   |

This could either be explained by the probe features becoming irrelevant and getting eroded as a side-effect of LayerNorm scaling, or because the model was trying to represent more features than it had dimensions. The latter would mean that there were more probes to discover, so I set out to find them! I came up with two theories for extra features that the model might be using.

#### Inductive logic

My first idea was that it could be working out board states inductively: layer by layer, it could take the current board, apply the next move, flip a bunch of tiles, and continue. In order to do so, it would use board state features corresponding to the $$previous$$ (PTEM) board state, as well as another feature corresponding to $$captured$$ (C) squares.

<!-- <img src="/assets/images/othello-gpt/probe_tem.png"> -->
<img src="/assets/images/othello-gpt/truth_tem.png">
<img src="/assets/images/othello-gpt/probe_ptem.png">
<img src="/assets/images/othello-gpt/probe_c.png">

The inductive logic for calculating a square's current state from the previous one would then be:

  - **(PT) -> (T)**: the opponent's squares cannot be captured by their own move
  - **(PM) + (C) -> (T)**: my captured squares get flipped
  - **(PM) + (~C) -> (M)**: my uncaptured squares stay mine
  - **(PE) -> (E) or (T)**: one previously empty square gets the opponent's move played on it

<div class="image-row">
<img src="/assets/images/othello-gpt/acc_ptem_c.png" width="350px">
<img src="/assets/images/othello-gpt/acc_ptem.png" width="350px">
</div>

Ignoring pos 0 in the accuracy calculations (I hoped that if I didn't prepend the initial board state to the training data, the probes would find the model's representation of it at pos 0, but this didn't work), the (PTEM) probes performed almost as well as the original (TEM) probes!

<!-- Now, we had three new probes: (PE), (PT-PM), and (C). However, the fact that (T-M) = (PT-PM) \|\| (C), which is equivalent to vector addition, meant that only one of these probes needed to be added to the basis. I decided on (C) due to its high accuracy. -->

#### Conditional logic

These inductive probes were cool, but I couldn't fit them into my picture of how OthelloGPT was computing the board state in just 5 layers. In order to do this, it had to be possible to calculate several sub-states in parallel. One obvious example was the $$empty$$ (E) state, which was computed after just one attention layer!

I figured that the model had to be separating out the simplest possible features for each square, computing each one in parallel, at the earliest layer possible, and then combining them in later layers. For example, the $$captured$$ (C) feature could be initialised as a maximum likelihood prior across all squares that a move could *potentially* capture and then refined over subsequent layers. This greedy approach would explain the better-than-random L0 probe accuracies.

The combination of these simple probes into useful outputs would then be done via conditional statements. For example, only $$empty$$ (E) squares can be $$legal$$ (L), and only (~E) squares can be $$theirs$$ or $$mine$$ (T-M). In a transformer, the two mechanisms available for these combinations are attention heads, which compute linear combinations across token positions, and neurons, which compute non-linear combinations within the same position.

<img src="/assets/images/othello-gpt/decision-flowchart.svg">

This idea led me down a rabbit-hole of training a lot of probes which were ultimately not very useful, but I think the underlying intuition is still nice! It suggests an intelligence paradigm more akin to how a highly parallelised, probabilistic machine brain would think, as opposed to a human one.

# Sprint

At this point, I was getting pretty bogged down in the project. I felt I hadn't really discovered anything concrete, the messy code was piling up, and my latest investigations had been frustratingly unfruitful. I decided to use the application process to Neel's MATS stream as a forcing function to get something done in a 16-hour sprint.

This made me choose an even narrower project scope. Instead of figuring out how the entire board state was computed, I decided to focus in on just the $$empty$$ (E) state, which turned out to be a good decision!

#### Revisiting the PE probe

<img src="/assets/images/othello-gpt/probe_ee.png">
<img src="/assets/images/othello-gpt/probe_pe.png">

In my earlier investigation, I trained a $$captured$$ (C) probe with the relationship (T-M) = (PT-PM) + (C). I hypothesised that a similar relationship could be found linking between (E) and (PE) - the difference between the two is equivalent to the move that was just played! Since both probes had high accuracies, I decided to see what happened if I just took the normed difference vector (PE-E) instead of training a new probe.

<div class="image-row">
<img src="/assets/images/othello-gpt/w_e_ee.png" width="300px"/>
<img src="/assets/images/othello-gpt/w_e_t_m.png" width="300px"/>
</div>
<div class="image-row">
<img src="/assets/images/othello-gpt/w_u_ee.png" width="300px"/>
<img src="/assets/images/othello-gpt/w_u_t_m.png" width="300px"/>
</div>

In my [previous post]({% post_url 2025-03-02-othello-gpt-0 %}), I found that the board state probes (EE) and (T-M) could be applied to the embedding (W_E) and unembedding (W_U) weights to generate interpretable images of how the model's token projections aligned with the features.

<img src="/assets/images/othello-gpt/w_e_pee_ee.png" width="300px">

Applying the (PE-E) vector to (W_E) produced a really clear image! This was sufficient information for computing (E) in one attention layer. All the model had to do was see which moves hadn't yet been played at each position and mark these as (E). And all I had to do was find an attention head circuit that did this.

#### Finding L2H5

I used two tools here to identify attention heads that were writing out (E) vectors. The first was Neel's TransformerLens library, which I used to cache the decomposed and cumulative residual stream vectors of each attention head across 200 forward passes of the model.

```python
input_ids = t.tensor(test_dataset["input_ids"][:200], device=device)
_, cache = model.run_with_cache(input_ids[:, :-1])
X, y_labels = cache.get_full_resid_decomposition(
    apply_ln=True, return_labels=True, expand_neurons=False
)
X /= X.norm(dim=-1, keepdim=True)
X_cum, y_cum_labels = cache.accumulated_resid(
    apply_ln=True,
    return_labels=True,
    incl_mid=True,
)
X_cum /= X_cum.norm(dim=-1, keepdim=True)
```

The second tool was using SVD to calculate the amount of variance across the batch that was explained by a given probe basis. I ran into some numerical instability when passing highly colinear probes into the function, so I added some checks for this.

```python
def calculate_explained_var(
    X: Float[t.Tensor, "batch ... d_model"],
    b: Float[t.Tensor, "d_model basis"],
):
    # Use double precision (not supported by mps) to minimise instability errors
    # Alternatively add a jitter term
    X = X.detach().cpu().double()
    b = b.detach().cpu().double()

    # Add a small regularisation term to b to perturb it
    b += 1e-6 * t.randn_like(b)

    # Perform SVD on the basis vectors to get an orthonormal basis
    U, S, _ = t.linalg.svd(b, full_matrices=False)

    # Calculate the conditioning number
    condition_number = S.max() / S.min()
    cond_threshold = 1e6
    if condition_number > cond_threshold:
        print(f"Condition number: {condition_number}")
        print(S / S.min())
        U = U[:, (S / S.min() > cond_threshold)]

    # Project X onto the orthonormal basis
    proj = t.matmul(X, U)

    # Reconstruct the projections
    proj_reconstructed = t.matmul(proj, U.transpose(-1, -2))

    # Compute the squared norms of the projections and the original vectors
    proj_sqnorm = t.sum(proj_reconstructed**2, dim=-1)
    X_sqnorm = t.sum(X**2, dim=-1)

    # Calculate the explained variance ratio
    explained_variance = proj_sqnorm / X_sqnorm

    # Take the average over the batch
    explained_variance = t.mean(explained_variance, dim=0)

    return explained_variance
```

I used ```calculate_explained_var``` to see how much of the residual contribution variance at each position was explained by the 32-dimensional (EE) basis, averaged across the batch.

<img src="/assets/images/othello-gpt/explained_var.png">

As expected, the L0 heads were fairly active in this basis, as were the L7 heads. I interpreted the latter as a computation in (U) space that spilled over into (EE) space due to colinearities from superposition, but investigating this was out of scope. Interestingly, L2H5 was the highest in explained variance, so I plotted some sample attention patterns using circuitsvis.

<div class="image-row">
<img src="/assets/images/othello-gpt/l2h5_0.png" width="200px"/>
<img src="/assets/images/othello-gpt/l2h5_1.png" width="200px"/>
<img src="/assets/images/othello-gpt/l2h5_2.png" width="200px"/>
</div>

It looked like L2H5 attended specifically to the D5 token, otherwise defaulting to pos 0. I re-ran the explained variance calculation with just the probe for the D5 square, instead of all 32 (EE) probes, and saw that this single vector accounted for over half of the output variance!

<img src="/assets/images/othello-gpt/explained_var_ee_D5.png">

#### Interpreting L2H5

Although I couldn't explain why a whole L2 attention head was seemingly being used to recompute the (EE) state for D5 (assuming that it had already been calculated after L0), I decided to go ahead with interpreting it anyway and deal with that question later. Firstly, I confirmed that the (PE-E) probe was able to extract the move played at each position from the residual stream at L2_pre by taking a cached residual stream from a sample game and transforming it using the probe.

<img src="/assets/images/othello-gpt/l2_resid_pee_ee.png">

Success! Now, with neurons, we saw in previous work that the (W_in) and (W_out) weights could be transformed into probe bases for interpretation. With attention heads, we can do the same with (W_Q), (W_K), (W_O), and (W_V). [Elaborate!]

I probed L2H5's (W_K) with (PE-E) and (P), expecting to see activations for (PE-E_27) and (P_0), corresponding to the D5 square and pos 0.

<div class="image-row">
<img src="/assets/images/othello-gpt/l2h5_w_k_pee_ee.png" width="300px">
<img src="/assets/images/othello-gpt/l2h5_w_k_p.png" width="300px">
<!-- <img src="/assets/images/othello-gpt/l2h5_w_q_p.png" width="300px"> -->
</div>

Almost all of L2H5's (W_K) head dimensions were dedicated to D5 in (PE-E) space! But W_K.P didn't show any clear pattern. Instead, I saw the expected pattern in (W_Q), in the same head dimensions where (PE-E_D5) was negatively aligned with (W_K). I didn't really understand the reason for this, since when querying from pos 0 it's only possible to self-attend due to causal masking...

Next, I probed L2H5's (W_V) with (PE-E) and (P), and also probed L2H5's (W_O) with (E), expecting to see activations in (W_V) for D5 and pos 0 and activations in (W_O) for D5.

# Conclusion

# Reflection
- Logging ideas, experiments, time tracking
- 
