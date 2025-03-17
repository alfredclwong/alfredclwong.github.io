---
layout: post
title: "[1] A deep dive into OthelloGPT: Sprint"
date: 2025-03-05 22:13:00 +0000
categories: jekyll update
---
<script type="text/javascript" id="MathJax-script" async
    src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
<link rel="stylesheet" href="/assets/css/style.css">

This post covers the content of a research sprint I did last month on OthelloGPT, following an initial [Replication]({% post_url 2025-03-02-othello-gpt-0 %}) project. I build on prior work by presenting a new linear feature in the model's residual stream, which I use to mechanistically interpret an attention head. Aside from some preliminary work on research directions that I decided not to pursue for the sprint, this work was done over 16 hours in 2 days.

# Table of Contents
<div style="border: 1px solid #ccc; background-color:rgb(239, 251, 255); padding: 10px; margin-bottom: 10px; width: 250px">
  <ul>
    <li><a href="#preliminary-work">Preliminary work</a>
      <ul>
        <li><a href="#status-quo">Status quo</a></li>
        <li><a href="#a-question-and-a-hunch">A question and a hunch</a></li>
        <li><a href="#inductive-theory">Inductive theory</a></li>
        <li><a href="#conditional-theory">Conditional theory</a></li>
      </ul>
    </li>
    <li><a href="#sprint">Sprint</a></li>
      <ul>
        <li><a href="#revisiting-the-pe-probe">Revisiting the PE probe</a></li>
        <li><a href="#finding-l2h5">Finding L2H5</a></li>
        <li><a href="#interpreting-l2h5">Interpreting L2H5</a></li>
      </ul>
    <li><a href="#improvements">Improvements</a></li>
    <li><a href="#conclusion">Conclusion</a></li>
  </ul>
</div>

# Preliminary work

#### Status quo

I picked up from the previous post replicating Kenneth Li and Neel Nanda with:

- A 6M param GPT2-style model (8 layers, 8 heads, 256 dimensions) that predicts legal next moves in 6x6 Othello with 99.97% accuracy.
  - The inputs and outputs of the model are purely textual, e.g. `C5 B3 E2 D5 B5 E5...`, with each of the 32 possible moves encoded as a unique token.
  - The model outputs a distribution across all 32 moves and we measure accuracy on the top-1 logit.
<img src="/assets/images/othello-gpt/truth.png">
<img src="/assets/images/othello-gpt/preds.png">

- A setup for training linear probes: matrices that map from OthelloGPT's residual stream to board state targets.
  - The residual stream consists of vectors $$\mathbf{x}_p^{(l)} \in \mathbb{R}^{256}$$ at each move position $$p \in \{0, \ldots, P-1\}$$ and layer $$l \in \{0, \ldots, L-1\}$$ that are produced by OthelloGPT at inference time.
  - At each position, we have information $$\mathbf{y}_p \in \{0, \ldots, K-1\}^{36}$$ on the state of each square on the board, e.g. whether a square is $$white$$ (0), $$empty$$ (1), or $$black$$ (2). We call these targets.
  - In a given game, we have a target matrix $$Y \in \{0, \ldots, K-1\}^{36 \times P}$$ and inputs $$X^{(l)} \in \mathbb{R}^{256 \times P}$$ for layer $$l$$. We use logistic regression to train linear maps $$M^{(l)}: X^{(l)} \mapsto \hat{Y}$$ from residual stream vectors to logits $$\hat{Y} \in \mathbb{R}^{36 \times P \times K}$$. We call this a linear probe for layer $$l$$ and we train a separate probe for each layer in the model, including "middle" layers in between the attention block and MLP, referred to as L0_mid or L0.5, etc.
  <!-- - I decided in the previous post to work with binary probes ($$K=2$$) as they represent the simplest possible features in the model. -->

- A binary linear probe (EE) that predicts whether a board square is $$empty$$ with 99.99% accuracy at L1_mid.
<img src="/assets/images/othello-gpt/probe_ee.png">

- A binary linear probe (T-M) that predicts whether a non-empty board square belongs to the opponent, i.e. is $$theirs$$ (T) and not $$mine$$ (M), with 99.1% accuracy at L5_mid.
<img src="/assets/images/othello-gpt/probe_tm.png">

- A setup for visualising vectors under the basis transformation defined by a binary linear probe.
  - For a binary probe $$M: \mathbb{R}^{256} \rightarrow \mathbb{R}^{36 \times 2}$$, I interpret the vector ```M[:, i, 1]``` as the direction in the residual stream space that aligns with the target feature for square $$i$$, where $$i=0$$ for ```A1```, $$i=1$$ for ```A2```, etc.
  - This allows us to transform any vector $$\mathbf{x} \in \mathbb{R}^{256}$$ into a 6x6 board image. For example, I previously transformed the input weights $$\mathbf{w}_i$$ and output weights $$\mathbf{w}_o$$ for neuron L5N415 to show that it activates strongly on a specific input board state to output a positive logit for a corresponding legal square.
<img src="/assets/images/othello-gpt/l5n415_in.png" height="120px"/>
<img src="/assets/images/othello-gpt/l5n415_out.png" height="120px"/>

  - I don't claim that this is the best way to define features, but it's a method with enough intuition and empirical success that I think it's worth running with for now and evaluating later.

#### A question and a hunch

I wanted to build on the work by finding something new, but this was also my first attempt at doing mech interp research. Rather than answering high level questions, such as "what makes a good feature?" or "how do we interpret superposition in transformers?", I went with a practical approach that lent itself more towards playful experimentation. I set out to answer the question: **"how does OthelloGPT compute its world model?"**

I had a hunch that the model was using additional latent features in order to compute the board state. There was some weak evidence for this:

- At the end of my previous post, I noticed alignment patterns between the board state probes (EE & T-M) and the unembedding vectors (U) that looked like superposition. I saw this by visualising W_U in the (EE) and (T-M) bases, showing for example that (U)_A1 was highly colinear with (EE)_A1, (T-M)_B2, and ~(T-M)_C3: board states that are highly correlated with but not equivalent to A1 being a legal move.

<div class="image-row">
  <img src="/assets/images/othello-gpt/w_u_ee.png" width="40%"/>
  <img src="/assets/images/othello-gpt/w_u_t_m.png" width="40%"/>
</div>

- The repertoire of feature vectors so far amounted to at most 163 dimensions out of 256 available.

| Feature vector         | Count |
|------------------------|-------|
| Token embed (B)        | 32    |
| Pos embed (P)          | 31    |
| Empty probe (EE)       | 32    |
| Theirs/mine probe (T-M)| 36    |
| Token unembed (U)      | 32    |
|------------------------|-------|
| **Total**              | 163   |

If this were the entire feature set, it would be possible for the model to represent these features as linearly separable monosemantic directions. The question is whether the model is sufficiently incentivised to do this. We've seen neurons which output legal logits using these board state features, so it seems that the superposition with (T-M) in particular could lead to some undesirable confounding.

An alternative explanation is that the model does all its legality computations in parallel, after which it no longer needs to maintain an accurate board state representation. Thus it's never necessary to represent the features simultaneously.

Either way, I decided that it would be useful to pursue the hunch and see if it was possible to find some interesting probes. I came up with two theories that could lead to additional features.

#### Inductive theory

My first idea was that OthelloGPT could be working out board states inductively: layer by layer, it could take the current board, apply the next move, flip a bunch of tiles, and continue. In order to do so, it would use board state features corresponding to the $$previous$$ (PTEM) board state, as well as another feature corresponding to $$captured$$ (C) squares.

<!-- <img src="/assets/images/othello-gpt/probe_tem.png"> -->
<img src="/assets/images/othello-gpt/truth_tem.png">
<img src="/assets/images/othello-gpt/probe_ptem.png">
<img src="/assets/images/othello-gpt/probe_c.png">

The inductive logic for calculating a square's current state from the previous one would then be:

  - **(PT) -> (T)**: the opponent's squares cannot be captured by their own move
  - **(PM) + (C) -> (T)**: my captured squares get flipped
  - **(PM) + ~(C) -> (M)**: my uncaptured squares stay mine
  - **(PE) -> (E) or (T)**: one previously empty square gets the opponent's move played on it

<div class="image-row">
<img src="/assets/images/othello-gpt/acc_ptem_c.png" width="350px">
<img src="/assets/images/othello-gpt/acc_ptem.png" width="350px">
</div>

Ignoring pos 0 in the accuracy calculations (I hoped that if I didn't prepend the initial board state to the training data, the probes would find the model's representation of it at pos 0, but this didn't work), the (PTEM) probes performed almost as well as the original (TEM) probes!

<!-- Now, we had three new probes: (PE), (PT-PM), and (C). However, the fact that (T-M) = (PT-PM) \|\| (C), which is equivalent to vector addition, meant that only one of these probes needed to be added to the basis. I decided on (C) due to its high accuracy. -->

#### Conditional theory

These inductive probes were cool, but I couldn't fit them into my picture of how OthelloGPT was computing the board state in just 5 layers. In order to do this, it had to be possible to calculate several sub-states in parallel. One obvious example was the $$empty$$ (E) state, which was computed across all game positions and board squares after just one attention layer!

I figured that the model had to be separating out the simplest possible features for each square, computing each one in parallel, at the earliest layer possible, and then combining them in later layers. For example, the $$captured$$ (C) feature could be initialised as a maximum likelihood prior across all squares that a move could *potentially* capture and then refined over subsequent layers (it's not possible to capture (E) squares, etc.). This greedy approach could explain the better-than-random probe accuracies at L0, immediately after embedding.

The combination of these simple probes into useful outputs would then be done via conditional statements. For example, only $$empty$$ (E) squares can be $$legal$$ (L), and only ~(E) squares can be $$theirs$$ or $$mine$$ (T-M).

<img src="/assets/images/othello-gpt/decision-flowchart.svg">

These combinations could be expressed as linear functions across token positions using attention heads, or as non-linear functions within the same position using neurons. For example, we previously saw that once the board state was computed at each position, neurons could be used to find legal moves. But in other cases, such as predicting the final move in a game, an attention head might be more suitable for finding all previous moves and outputting the remaining empty square as the only possible move.

This idea led me down a rabbit-hole of training a lot of probes which were ultimately not very useful, but I think the underlying intuition is still nice! It suggests an intelligence paradigm more akin to how a highly parallelised, probabilistic machine brain would think, as opposed to a human one.

# Sprint

At this point, I was getting pretty bogged down in the project. I felt I hadn't really discovered anything concrete, the messy code was piling up, and my latest investigations had been frustratingly unfruitful. I decided to use the application process to Neel's MATS stream as a forcing function to get something done in a 16-hour sprint.

This made me choose an even narrower project scope. Instead of figuring out how the entire board state was computed, I decided to focus in on just the $$empty$$ (E) state.

#### Revisiting the PE probe

While investigating my [Inductive Theory](#inductive-theory), I trained a $$captured$$ (C) probe with the relationship (T-M) = (PT-PM) + (C). I hypothesised that a similar relationship could be found linking (E) and (PE) - the difference between the two is equivalent to the move that was just played! Similarly to before, I trained binary probes (EE) and (PEE) that ignored which player the non-empty squares belonged to. This time round, the performance between the existing (PE) probe and new (PEE) probe was identical.

```python
def empty_target(batch, device):
    boards = t.tensor(batch["boards"], device=device)[:, :-1]
    return (boards == 0).flatten(2)

def prev_empty_target(batch, device, n_shift=1):
    e = empty_target(batch, device)
    n_batch = e.shape[0]
    n_out = e.shape[-1]
    e0 = t.full((n_batch, n_shift, n_out), t.nan, device=device)
    return t.cat([e0, e[:, :-n_shift]], dim=1)
```

<img src="/assets/images/othello-gpt/probe_ee.png">
<img src="/assets/images/othello-gpt/probe_pe.png">
<img src="/assets/images/othello-gpt/acc_pee.png">

Since both probes had high accuracies, I was a bit lazy and just worked with the normed difference vector (PEE-EE) instead of training a new probe. Using this (PEE-EE) basis as a transformation for the embedding weights (W_E) showed that the probe could indeed extract the move that each token embedding was representing!

<img src="/assets/images/othello-gpt/w_e_pee_ee.png" width="300px">

This was sufficient information for computing the (EE) board state using just one layer: all the model had to do was see which moves hadn't yet been played at each position and mark these as (EE). And all I had to do was find the attention heads that did this...

#### Finding L2H5

I used two tools here to identify attention heads that were writing out (EE) vectors. The first was Neel's TransformerLens library, which I used to cache the decomposed residual stream vectors being written by each attention head across 200 forward passes of the model.

```python
input_ids = t.tensor(test_dataset["input_ids"][:200], device=device)
_, cache = model.run_with_cache(input_ids[:, :-1])
X, y_labels = cache.get_full_resid_decomposition(
    apply_ln=True, return_labels=True, expand_neurons=False
)
X /= X.norm(dim=-1, keepdim=True)
```

The second tool was an SVD calculation to find the amount of variance that was explained by a given probe basis. I ran into numerical instability issues when passing highly colinear probes into the function, so I added some checks for this.

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

I used these tools to visualise how much of the decomposed residual stream at each position was explained by the 32-dimensional (EE) basis, averaged across the batch.

<img src="/assets/images/othello-gpt/explained_var.png">

As expected, the L0 heads were fairly active in this basis, as were the L7 heads. I interpreted the latter as a computation in (U) space that spilled over into (EE) space due to colinearities, but investigating this was out of scope. Interestingly, L2H5 was the highest in explained variance, so I looked at some sample attention patterns using circuitsvis.

<div class="image-row">
<img src="/assets/images/othello-gpt/l2h5_0.png" width="30%"/>
<img src="/assets/images/othello-gpt/l2h5_1.png" width="30%"/>
<img src="/assets/images/othello-gpt/l2h5_2.png" width="30%"/>
</div>

It looked like L2H5 attended specifically to the D5 token, otherwise defaulting to pos 0. I re-ran the explained variance calculation with just the probe for the D5 square, instead of all 32 (EE) probes, and saw that this single vector accounted for over half of the output variance!

<img src="/assets/images/othello-gpt/explained_var_ee_D5.png">

#### Interpreting L2H5

Although I couldn't explain why a whole L2 attention head was seemingly being used to recompute the (EE) state for D5 (assuming that it had already been calculated after L0), I decided to go ahead with interpreting it anyway and deal with that question later.

Firstly, I checked whether the (PEE-EE) probe was still able to extract the latest move from each position at L2_pre by taking a cached game and transforming the residual vector at each position using the probe.

<img src="/assets/images/othello-gpt/l2_resid_pee_ee.png">

Success! Recall that we previously interpreted neurons by transforming their input and output weights into probe bases. Interpreting attention heads required a slightly different approach. Following Anthropic's [mathematical framework](https://transformer-circuits.pub/2021/framework/index.html), attention heads can be separated into two bilinear forms (W_QK) and (W_OV), where

$$A = softmax(\mathbf{x_{dst}} W_Q^T W_K \mathbf{x_{src}})$$

$$\mathbf{x_{out}} = (A \otimes W_O W_V) \cdot \mathbf{x_{src}}$$

Thus, it was possible to probe each individal weight matrix to see how they aligned with certain features across head dimensions. Interpreting these results was more difficult than with the neurons because there were multiple head dimensions, compared to the single scalar activation value for each neuron, so it was possible that linear combinations were being taken across head dimensions. Fortunately, L2H5 was almost as monosemantic as attention heads can get.

I probed L2H5's (W_K) with (PEE-EE) and the $$positional$$ (P) embedding, expecting to see strong alignment with (PEE-EE)_D5 and (P)_0. Note: the (P) image is arranged into a board image but in this case it doesn't mean anything.

<div class="image-row">
<img src="/assets/images/othello-gpt/l2h5_w_k_pee_ee.png" width="45%">
<img src="/assets/images/othello-gpt/l2h5_w_k_p.png" width="45%">
</div>

The (PEE-EE) images were clearly well aligned with D5, but (P) was harder to interpret. It looked like the dimensions where (PEE-EE) aligned negatively with (W_K), it aligned well with a range of positions - mostly pos 0/1/3.

Next, I probed L2H5's (W_V) with (PEE-EE) and (P), and also probed L2H5's (W_O) with (EE), expecting to see activations in (W_V) for D5 and pos 0 and activations in (W_O) for D5.

<div class="image-row">
<img src="/assets/images/othello-gpt/l2h5_w_v_pee_ee.png" width="45%">
<img src="/assets/images/othello-gpt/l2h5_w_v_p.png" width="45%">
</div>

The images for (W_V) weren't super clear, but it was more important to see which (W_O) head dimensions were dedicated to writing out to (EE)_D5 first:

<img src="/assets/images/othello-gpt/l2h5_w_o_ee.png" width="45%">

From the (W_O) transformations, I could see that dimensions 19 & 27 wrote out (EE)_D5 and dimensions 20 & 23 wrote out ~(EE)_D5. Referring back to the (W_V) images, the (EE)_D5 dimensions roughly matched to (P)_0 and ~(PEE-EE)_D5 source tokens, and the ~(EE)_D5 dimensions matched to ~(P)_0 and (PEE-EE)_D5!

Putting everything together, I showed that:

  - (W_K) aligned strongly with (PEE-EE)_D5 and (P)_0/1/3 such that L2H5 would attend to either the position where D5 was played or pos 0/1/3 if D5 hadn't yet been played
  - (W_OV) would write out (EE)_D5 if L2H5 attended to (P)_0 or ~(EE)_D5 if it attended to ~(P)_0!

# Improvements

The sprint was completed under pretty tight time constraints, so a lot of the decisions I made were suboptimal. Here, I make a few quick clean-ups that improve the quality of the results.

#### (MOV) probe

I pretty much based the investigation around the (PEE-EE) "probe" and luckily managed to do pretty well. However, it's actually pretty poor at predicting the $$latest\ move$$ (MOV). It's possible to train a probe for this target with pretty much 100% accuracy.

<img src="/assets/images/othello-gpt/acc_mov.png">

#### Bilinear probe visualisation

Interpreting the L2H5 attention head across all head dimensions was pretty clunky because I looked at each weight matrix individually. Instead, we can apply the probes to each side of the full bilinear form.

<div class="image-row">
<img src="/assets/images/othello-gpt/l2h5_ee_ov_mov.png" width="45%">
<img src="/assets/images/othello-gpt/l2h5_ee_ov_p.png" width="45%">
</div>

This is much more interpretable. In fact, we can see now that, while the (MOV)_D5 source tokens write out ~(EE)_D5, the (P)_0 source tokens don't actually output (EE)_D5!!

#### (P) basis reduction

None of the images generated by (P) transforms were particularly interpretable. In the previous post, we saw that there was a lot of colinearity between the 31 (P) vectors. This means that there might be a lower dimension basis that is better to work with. We perform this dimensionality reduction using PCA and plot the alignment patterns of the top 10 resulting (PR) vectors, which account for 90% of (P) variance, with the original (P) basis.

<img src="/assets/images/othello-gpt/pca_p.png">
<img src="/assets/images/othello-gpt/pr_p.png">

Now, let's see if this produces a more interpretable image in the bilinear visualisation.

<img src="/assets/images/othello-gpt/l2h5_ee_ov_pr.png">

It looks like (PR)_0, which aligns with (P)_0 and (P)_1, writes out (EE)_D5. This isn't a particularly strong result but could be worth further investigating.

# Conclusion

My goal for the project was to find evidence for additional probes and use them to interpret an attention head for computing the board state. I think I managed to achieve this!

  - I found two interesting probes, corresponding to squares that were just $$captured$$ (C) and just $$moved$$ (MOV). The inductive probes (PT-PM) and (PEE) can be derived as linear combinations: (PT-PM) = (T-M) - (C) - (MOV) and (PEE) = (EE) + (MOV).
  - I used cached residual stream contributions from each attention head to identify L2H5, which almost exclusively outputted (EE)_D5 vectors.
  - I showed that the QK circuit in L2H5 attended to D5 if it had been previously played, and then showed that the OV circuit wrote out that D5 was not empty if attended to.

In terms of further work along this direction, I think it would be cool to interpret a head that computes captures. I would imagine it works by using (MOV) to attend to all previously played capturable squares and using (T-M) to identify which ones belong to the other player. I could also imagine a head that memorises common openings. For example, if a square far from the centre is played at a relatively early position, there are only a limited number of game trees that could have made this possible, which the head could categorise according to past moves.

As for implications for the wider field of mech interp, this work was different to existing work because it focused on attention heads rather than neurons and tried to find intermediate features with little direct relevance to the output logits. All operations within attention heads are linear, so linear probes were highly suited to the task.

I think it would be interesting to do some follow-ups on how these features could be identified in an unsupervised manner. I was mostly guided by intuition and probe accuracy, and then became more confident in the probe validity as interpretable images were generated by transforming weight matrices and residual vectors. I generally defined "interpretability" as having sparse activations, implying monosemanticity. The intuition was mostly directed at how the model might generate intermediate features for use in later calculations. The following is a rough list of "things which might make a good feature":

  - Intuitive meaning/purpose
  - High linear probe accuracy
  - Sparse alignment with weight matrices
  - High alignment with forward-pass residual stream vectors
  - Causal interventions?

Performing a literature review after my sprint (the other way round definitely makes more sense), I found a paper on [Sparse Dictionary Learning](https://arxiv.org/pdf/2402.12201) that seemed to systemise this into an unsupervised method for identifying features. If I work on this problem further, I'd like to see if I can build on this.
