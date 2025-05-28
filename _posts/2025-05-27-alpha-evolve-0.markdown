---
layout: post
title: "[Draft] Can we use AIs to write AIs?"
date: 2025-05-27 15:25:00 +0000
---
<script type="text/javascript" id="MathJax-script" async
    src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
<link rel="stylesheet" href="/assets/css/style.css">

It's been 2.5 years since ChatGPT was released to the public, and the AI product cycle seems to have settled into a rhythm. Every few months, a leading lab releases their latest, greatest model, benchmarks are broken, your favourite [YouTuber](https://www.youtube.com/@Fireship) vibe tests it, we conclude that jobs still exist, and then we go back to said jobs and wait for the next one.

No-one really knows where this is headed. Scaling Laws are the new Moore's Law, and some say that AGI is only a factor of FLOPs over the horizon. On the other hand, naysayers maintain that LLMs are just fancy-dressed copy-cat statistical surfaces posing as faux-intelligentsia.

Amidst all the debate, Alhussein Fawzi and Bernardino Romera Paredes decided that, if monkeys can type Shakespeare, then maybe LLMs can do science. Enter [FunSearch](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/): an evolutionary framework developed at DeepMind that uses LLMs to generate programmatic solutions to fixed tasks.

{% include image.html url="https://lh3.googleusercontent.com/mIlL5zg4-gbYvIdCuBB5SmzPzbC1yUghYgIwYR89pEJpgc4f00OhDpd6SRM_MXNi1XSqJJpFe_yeFXHZShLr3syM0SFSxtuqJzdaEgX8fsvCW1SN=w616-rw" description="The FunSearch process" %}

The framework centres around solving a task, e.g. [the bin-packing problem](https://en.wikipedia.org/wiki/Bin_packing_problem), where solutions can be quantitatively evaluated. An LLM is prompted to generate solutions in the form of programs, which are scored and stored in a database, which in turn is then sampled to form a positive feedback loop for the next prompt and generation.

FunSearch is good at solving combinatorial problems, often using a human-provided code skeleton such that all it has to do is come up with a heuristic function for choosing between many options. Below is a simple example of a heuristic function that it came up with for bin packing.

<embed src="/assets/html/bin_packing_demo.html" style="width:100%; height: 1300px; justify-content: center;">

It's so good at this, in fact, that it managed to find a new best [solution](https://colab.research.google.com/github/google-deepmind/funsearch/blob/master/cap_set/cap_set.ipynb) to the cap-set problem. This is pretty exciting stuff, as it's one of the few current examples of LLM-generated scientific discovery. It also presents a fun engineering challenge, as the evolutionary algorithm benefits from the scalability of highly parallelised processing. For these reasons, I decided it would make a suitable candidate for my next paper replication.

# Table of Contents
* TOC
{:toc}

# Replication

The core components of the algorithm are:
- Task
- Evaluation
- Database
- Prompt

## Task

Since DeepMind already tackled mathematics, I chose to apply the algorithm to the task of developing board game AIs, specifically for [Othello](https://www.worldothello.org/about/about-othello/othello-rules/official-rules/english). I asked for completions to the following function, using the board state, current player, and time remaining (starting at 9999ms for each player) to return a chosen move.

```python
def ai(board: T_BOARD, player: Player, clock: T_CLOCK) -> T_SQUARE:
    """
    AI function to select a move in the Othello game.

    Args:
        board (T_BOARD): The current state of the Othello board. board[:, :, 0] represents player 0's pieces,
                         and board[:, :, 1] represents player 1's pieces.
        player (Player): The current player (value 0 or 1).
        clock (T_CLOCK): Remaining time for both players in milliseconds.

    Returns:
        T_SQUARE: The selected move as a tuple of coordinates (x, y).
    """
```

I could have provided more of a skeleton here, for example by setting up a minimax search tree and asking for a scoring function, but I was curious about how the algorithm would perform with a more open-ended task.

## Evaluation

Each completion was run in a Docker container against a collection of preset AIs, following an [evaluation cascade](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf) where a 90% win-rate was required to advance to the next AI. The levels of difficulty were:

1. Random: select a random legal move
2. Greedy: select the move which flips the most pieces
3. [Egaouricid4](https://github.com/Nyanyan/Egaroucid4): a 4 year old previously SOTA Othello AI, set to various search depths

Each level consisted of 100 games (split 50/50 between black/white) where each win was scored +1, each loss -1, and each draw 0. At each level, the minimum possible score was 0.

## Database

In anticipation of asynchronous processing, I stored the completions and scores in separate tables. I also tracked the tree structure of which completions were used as inspirations for later completions.

## Prompt

```
Act as an expert Python developer. Your job is to provide the best possible completion of \
a code skeleton. The completion will be appended to the skeleton and the completed code \
will be scored on its ability to complete a task. Higher scores are better.\

<TASK>
...
</TASK>
<SKELETON>
...
</SKELETON>

Previously we found that the following completions scored highly. You should aim to achieve \
a higher score by making improvements and/or trying new ideas.\

<COMPLETION_ID>
...
</COMPLETION_ID>
<SCORE_ID>...</SCORE_ID>
...
<COMPLETION_ID>
...
</COMPLETION_ID>
<SCORE_ID>...</SCORE_ID>

Output your reasoning in between <REASONING> and </REASONING> tags, followed by the code \
completion in between <COMPLETION> and </COMPLETION> tags. Do not output any other text. \
The reasoning should explain how this completion will improve upon previous iterations. \
The completion will be appended to the skeleton into a single function, so it should not \
repeat the function signature and it should start with one level of indentation. If you \
import libraries or define helper functions, make sure to do so within the scope of the \
function and before they are used. Make sure that the completion is valid Python code.\
"""
```

I used gemma-3-27b-it for free on OpenRouter, asking for reasoning in the prompt as well as the code completion. The inspirations were sampled as the topk scoring completions in the database. With 3 inspirations per prompt, the number of input tokens grew to about 6000, and the output tokens were limited to 2000. The generation speed was around 50-60 tps.

# Results
## The Good

{% include image.html url="/assets/images/alpha-evolve/meme.png" %}
This was a super basic replication on a fairly ambitious task, so I wasn't expecting much. I was pleasantly surprised to see that, after an overnight run, a sufficiently smart AI had emerged that could beat Egaroucid on a search depth of 2!

<embed src="/assets/html/evolutionary_tree.html" style="width:100%; height:1520px; justify-content: center;">

ai_277 scores each move using various sub-scores, ranging from corner control and number of flips to more obscure definitions such as 'stability' and 'mobility'. The total score is a weighted sum, and the weights are also a function of the game_stage, a variable which is (probably mistakenly) calculated as the number of player-owned pieces on the board.

## The Bad

The evolutionary steps seemed to vary between tweaking the weights and introducing/removing sub-scores, with the overall structure fairly constant after the first few generations. This meant that the trajectory was extremely prone to plateauing, as the evolutions randomly searched the weight space, or even degenerating, as important sub-scores were removed. In reality, I had to re-initialise the database if I saw that a stale path was being followed.

## The Ugly

Even worse, since nothing was done to encourage branching, an unlucky start to the evolutionary tree could result in a dead-end. For example, one run began by implementing a greedy algorithm that selected the move which flipped the most tiles. Since this number is non-negative, this implementation found the move with the highest score by initialising `max_flips = -1` and then iterating over the scores. This subsequently evolved into a score-based approach which ended up including negative components, but now with `max_score = -1` rather than `-float('inf')`, as a legacy of its evolutionary roots. Thus, the AI would return `None` when it assigned negative scores to all valid moves, resulting in a loss due to illegal play.

```python
def ai_1(board: T_BOARD, player: Player, clock: T_CLOCK) -> T_SQUARE:
    valid_moves = get_legal_squares(board, player)
    if not valid_moves:
        return None  # No valid moves available

    best_move = None
    max_flips = -1

    for move in valid_moves:
        flips = get_flips(board, player, move)
        num_flips = len(flips)

        if num_flips > max_flips:
            max_flips = num_flips
            best_move = move

    return best_move
```

```python
def ai_4(board: T_BOARD, player: Player, clock: T_CLOCK) -> T_SQUARE:
    valid_moves = get_legal_squares(board, player)
    if not valid_moves:
        return None  # No valid moves available

    size = get_size(board)

    # Prioritize corner squares
    corner_moves = [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]
    valid_corner_moves = [move for move in corner_moves if move in valid_moves]

    if valid_corner_moves:
        return valid_corner_moves[0]

    # Prioritize edge squares
    edge_moves = []
    for r in range(size):
        edge_moves.append((r, 0))
        edge_moves.append((r, size - 1))
    for c in range(1, size - 1):
        edge_moves.append((0, c))
        edge_moves.append((size - 1, c))

    valid_edge_moves = [move for move in edge_moves if move in valid_moves]

    if valid_edge_moves:
        return valid_edge_moves[0]

    # Evaluate moves based on flips and mobility
    best_move = None
    max_score = -1

    for move in valid_moves:
        flips = get_flips(board, player, move)
        num_flips = len(flips)

        # Calculate mobility after the move
        temp_board = board.copy()
        for r, c in flips:
            temp_board[r, c, player.value] = True
            temp_board[r, c, (~player).value] = False
        temp_board[move] = True  # Place the current player's piece
        
        next_player = (~player)
        next_valid_moves = get_legal_squares(temp_board, next_player)
        mobility = len(next_valid_moves)

        # Calculate a score based on flips and mobility
        score = num_flips + 0.5 * mobility  # Weight flips more heavily

        if score > max_score:
            max_score = score
            best_move = move

    # If time is running low, return a random valid move
    if clock[player.value] < 100:
        return valid_moves[np.random.randint(len(valid_moves))]

    return best_move
```

```python
def ai_72(board: T_BOARD, player: Player, clock: T_CLOCK) -> T_SQUARE:
    valid_moves = get_legal_squares(board, player)

    # ...

    # Evaluate moves based on flips and future legal moves
    best_move = None
    max_score = -1

    for move in valid_moves:
        flips = get_flips(board, player, move)
        num_flips = len(flips)

        # Create a temporary board to evaluate future moves
        temp_board = board.copy()
        for r, c in flips:
            temp_board[r, c, player.value] = True
            temp_board[r, c, (~player).value] = False
        temp_board[move] = True

        # Count the number of legal moves for the current player after the move
        future_legal_moves = len(get_legal_squares(temp_board, player))

        # Check if the move gives the opponent a corner in the next turn
        opponent_can_take_corner = False
        for corner in [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]:
            if is_empty(temp_board, corner) and is_legal_square(temp_board, (~player), corner):
                opponent_can_take_corner = True
                break

        # Calculate a score based on flips and future legal moves
        score = num_flips * 10 + future_legal_moves - (5 if opponent_can_take_corner else 0)  # Weight flips more heavily

        if score > max_score:
            max_score = score
            best_move = move

    # If time is running low, return a random valid move
    if clock[player.value] < 5:
        return valid_moves[np.random.randint(len(valid_moves))]

    return best_move

```

# Improvements
Basically follow the AlphaEvolve improvements, ranked by ablation effect

## Evolutionary prompting
- Multiple, stochastic templates (tweak/fix/prune/explore)
- Probabilistic inspiration sampling (MAP elites)
- Richer context (evolution progress, more metrics)
## Faster processing
- Parallel evals
  - More evals to reduce randomness/discrete jumps
  - More metrics
- Parallel generations
  - Remove reasoning and code comments
  - [maybe] Diff model
