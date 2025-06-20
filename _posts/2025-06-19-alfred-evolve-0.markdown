---
layout: post
title: "[Draft] Replicating AlphaEvolve"
date: 2025-06-19 17:50:00 +0100
---
<script type="text/javascript" id="MathJax-script" async
    src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
<link rel="stylesheet" href="/assets/css/style.css">

A couple months ago, GDM released [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/), their latest update on LLM-driven code generation using evolutionary methods. In their [paper](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf), they detailed how a few tweaks to their previous iteration, called [FunSearch](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/), created a much more powerful agent. They demonstrated this by applying AlphaEvolve to a bunch of maths and engineering problems and coming up with new best solutions in many cases.

{% include image.html url="/assets/images/alpha-evolve/alphaevolve.png" description="AlphaEvolve pipeline" %}

With powerful and fast LLMs such as [Gemini 2.5 Flash](https://openrouter.ai/google/gemini-2.5-flash-preview-05-20) accessible for only a few cents per million tokens, and the simplicity of verifying some of the mathematical tasks, this looked like a great candidate for at-home replication. In fact, someone had already created a cool open-source replication called [OpenEvolve](https://github.com/codelion/openevolve/tree/main)!

However, OpenEvolve's results were somewhat lacklustre: in replicating the [circle-packing problem](https://github.com/codelion/openevolve/tree/main/examples/circle_packing), they used scaffolding in the form of an initial program and a two-stage process with manual changes to the settings and task-specific prompts.

{% include image.html url="/assets/images/alpha-evolve/openevolve_prompt2.png" description="OpenEvolve's second-stage prompt configuration" %}

This didn't sit right with me: I subscribe to the view that advanced AIs shouldn't need human hand-holding to achieve results, à la David Silver's and Richard Sutton's ["Era of Experience"](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf) thesis (I highly recommend this [podcast episode](https://youtu.be/zzXyPGEtseI?si=qW4ek0_13GIvt_6M) on the topic). So, I looked into what it would take to Do It Myself.

{% include image.html url="/assets/images/alpha-evolve/vs_funsearch.png" description="FunSearch vs AlphaEvolve" %}

I referenced the above table from the AlphaEvolve paper to see what the main engineering challenges would be, and summed them up as follows:

1. Storing more data to enrich prompt content
2. Using diff-based evolutionary steps to support more complex programs
3. Exploiting parallelism to increase throughput while maintaining program diversity

There were lots of aspects to this project that excited me: I like maths, and I'd recently dabbled in an agentic coding project involving code diffs, parallelised processing, and databases for LLM training. I figured, what the hell, let's slap my name on it and try to make my own replication with cleaner prompting. Thus was born [AlfredEvolve](https://github.com/alfredclwong/alfred-evolve).

## Replication
### Pipeline
AlphaEvolve consists of an evolution loop that iterates through the following steps:

1. Sample a parent program and inspiration programs from the program database
2. Build a prompt using the sampled programs
3. Generate a diff by passing the prompt to an LLM
4. Apply the diff to the parent to get a child program
5. Evaluate the child program to get scores
6. Store the results

I decided to add reasoning to the diff generation step in order to further enrich the prompts, and also allowed the program evaluation to return artifacts, such as the exact circle packing generated by a program, for posterity. This resulted in the following initial code skeleton:

```python
parent, inspirations = program_database.sample()

# Steps 2-5 of the AlphaEvolve pipeline
prompt = build_prompt(parent, inspirations, cfg.prompt_builder_cfg)
diff, reasoning = generate_diff_and_reasoning(prompt, api_key, cfg.llm_cfg)
child_content = apply_diff(parent.content, diff)
scores, artifacts = evaluate_program(child_content, cfg.program_evaluator_cfg)

program_database.add_program(result)
```

Steps 1 & 6 involve database operations, which need careful handling in parallelised pipelines, so I set them aside initially and just implemented the intermediate core steps.

#### Prompt building
Each prompt consists of several parts:
- Preamble: a high-level description of the AlphaEvolve process
- Task: a specific description of the problem to be solved, e.g. circle packing
- Parent: a multi-part section consisting of the parent code and its associated reasoning and scores
- Inspirations: similar to the parent section but for several inspiration programs
- Variation: a randomly sampled prompt variation, e.g. encouraging this diff to refactor the parent program without changing core functionality, or to write a completely new program from scratch
- Epilogue: a description of the specific format expected from the output

Both the variations and the program sampling are used to provide an element of stochasticity to the evolution process in order to maintain diversity.

```
<PREAMBLE>
Act as an expert Python developer. Your job is to make iterative improvements to a source file in order to score highly on a task. You will be provided with a task description, a parent program, and a set of inspiration programs, which are previous attempts at solving the task. Your output will be a diff that will be applied to the parent program to create a new program.
</PREAMBLE>
<TASK>
You are an expert mathematician specializing in circle packing problems and computational geometry. Your task is to improve a constructor function that produces a specific arrangement of 26 circles in a unit square, such that none of them overlap. The function `pack_26()` should return a numpy array with 26 (x, y, r) rows, where (x, y) is the center of a circle and r is its radius. The score will be the sum of the radii of all circles, which you should maximise. Invalid packings, where circles overlap or extend beyond the unit square, will score 0. Functions which take more than 600 seconds to run will time out and score 0. The code for checking overlaps and bounds works with a numerical tolerance of 1e-9. This is a difficult problem so hard-coded solutions will not work well. The current best score found by other researchers is 2.635. The Python environment has the following libraries available: numpy, scipy.
</TASK>
<PARENT>
...
</PARENT>
<PARENT_REASONING>
* The previous attempts have not yielded any valid packing. This suggests that a complete rewrite is necessary.
* This new approach attempts to place circles in a structured, grid-like manner, which is a common strategy for circle packing.
* It starts with a central large circle and then tries to pack smaller circles around it and in the remaining space.
* The use of a `while` loop and `np.random.rand` suggests an iterative approach, potentially allowing for some randomized exploration of the packing space.
* The `is_valid` function (though not fully implemented in this snippet) indicates an intention to check for overlaps and bounds, which is crucial for a valid packing.
* The current `pack_26` implementation is a placeholder, and the goal is to provide a more sophisticated strategy.
</PARENT_REASONING>
<PARENT_SCORES>
{'INVALID_CODE_CHECK': 1.0, 'TIMEOUT_CHECK': 1.0, 'INVALID_TYPE_CHECK': 1.0, 'INVALID_LENGTH_CHECK': 1.0, 'INVALID_CIRCLE_CHECK': 1.0, 'OUT_OF_BOUNDS_CHECK': 1.0, 'OVERLAP_CHECK': 1.0, 'VALID_CHECK': 1.0, 'SCORE': 0.9674398198089572}
</PARENT_SCORES>
<INSPIRATION_57>
...
</INSPIRATION_57>
<INSPIRATION_57_REASONING>
...
</INSPIRATION_57_REASONING>
<INSPIRATION_57_SCORES>
...
</INSPIRATION_57_SCORES>
<INSPIRATION_51>
...
</INSPIRATION_51>
<INSPIRATION_51_REASONING>
...
</INSPIRATION_51_REASONING>
<INSPIRATION_51_SCORES>
...
</INSPIRATION_51_SCORES>
<VARIATION>
Try to tweak the parameters used in the previous completions to improve the score. This might include changing weights, adjusting learning rates, or modifying other hyperparameters. The goal is to find a better configuration that leads to a higher score without changing the overall approach or algorithm significantly.
</VARIATION>
<EPILOGUE>
Your output should consist of two parts: your reasoning for the changes and the diff itself. The reasoning should be a concise bullet-point list of the reasons why you believe the diff will improve the program's score. The diff should consist of SEARCH/REPLACE blocks that can be applied to the parent, and no other text. One diff may contain multiple SEARCH/REPLACE blocks, separated by newlines. The resulting program should be a valid Python program that will attempt to solve the task. It is important that the diff and code are valid, as invalid outputs will waste resources and time. Your response is limited to a maximum of 8192 tokens, so your changes must be small and focused.

SEARCH/REPLACE block rules:
1. Each SEARCH/REPLACE block consists of a SEARCH section and a REPLACE section
2. The SEARCH section begins with `<<<<<<<< SEARCH`, the REPLACE section begins with `========`, and the end of the block is marked with `>>>>>>>> REPLACE`.
3. The SEARCH section contains the code to be replaced, which should be uniquely identifiable within the parent program.
4. The REPLACE section contains the code that should replace the SEARCH code.
5. Both sections operate on a line-by-line basis.
6. A special case is when the SEARCH section is an empty line, which means the entire parent program should be replaced with the REPLACE section.

Example #1:
<PARENT>
def main():
    print("Hello, world!")
</PARENT>
<DIFF>
<<<<<<<< SEARCH
    print("Hello, world!")
========
    print("Aloha, world!")
>>>>>>>> REPLACE
</DIFF>

Example #2:
<PARENT>
def main():
    print("Hello, world!")
</PARENT>
<DIFF>
<<<<<<<< SEARCH

========
if __name__ == "__main__":
    print("Aloha, world!")
>>>>>>>> REPLACE

Your output should be formatted as follows:

<REASONING>{reasoning}</REASONING>
<DIFF>{diff}</DIFF>

</EPILOGUE>
```

#### Diff/reasoning generation
I used a POST request to query OpenRouter's API and then parsed the result by searching for the `<REASONING>` and `<DIFF>` tags.

#### Program evaluation
In order to safeguard against malicious code, I evaluated the programs in separate Docker containers. This was done by copying a task-specific evaluation script as well as the program code onto a container, executing the script, then parsing the output to retrieve scores and artifacts.

```python
def evaluate_program(
    program_content: str, cfg: ProgramEvaluatorConfig
) -> tuple[dict[str, float], dict[str, str]]:
    name = docker_utils.start(
        base_name=cfg.base_name,
        image=cfg.image,
        memory_limit=cfg.memory_limit,
        cpu_limit=cfg.cpu_limit,
    )

    try:
        return docker_utils.run(
            container_name=name,
            program_content=program_content,
            eval_file_path=cfg.eval_file_path,
            timeout=cfg.timeout,
        )
    except Exception:
        raise
    finally:
        docker_utils.stop(name=name)

# docker_utils.run
def run(
    program_content: str, container_name: str, eval_file_path: Path, timeout: int
) -> tuple[dict[str, float], dict[str, str]]:
    eval_result = _eval(container_name, program_content, eval_file_path, timeout)
    score_str = extract_tagged_text(eval_result, "SCORE")
    score_dict = parse_json(score_str)
    score_dict = {k: float(v) for k, v in score_dict.items()}
    artifacts_str = extract_tagged_text(eval_result, "ARTIFACT")
    artifacts_dict = parse_json(artifacts_str)
    return score_dict, artifacts_dict
```

### Data model
I've learnt over the course of several coding projects that things quickly spiral out of control if you don't keep the data structures tidy, especially so when parallelism is involved. My initial implementation had separate classes for Programs, Prompts, Diffs, and Results, but I found that this was much more manageable in one core Program class.

A Program has the following qualities:

- Each Program belongs to an evolutionary tree of Programs
- A child Program is created by applying a diff to a parent Program
- Each child Program can be inspired by several previous Programs
- Each step in the pipeline creates an object that is associated with the child Program

I created a data model to support this tree structure by assigning each Program an `id` and tracking its `parent_id` and `inspired_by_ids`. This was translated into an SQL database using sqlalchemy's ORM library, where the many-to-many inspiration relationships are stored in a separate table.

```python
@dataclass
class Program:
    id: Optional[int]
    content: str
    parent_id: Optional[int] = None
    inspired_by_ids: Optional[list[int]] = None
    prompt: Optional[str] = None
    reasoning: Optional[str] = None
    diff: Optional[str] = None
    scores: Optional[dict[str, float]] = None
    artifacts: Optional[dict[str, str]] = None
```

```python
Base = declarative_base()

inspiration = Table(
    "inspiration",
    Base.metadata,
    Column("inspired_by_id", ForeignKey("program.id"), primary_key=True),
    Column("inspired_id", ForeignKey("program.id"), primary_key=True),
)

class ProgramModel(Base):
    __tablename__ = "program"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # ...
    parent_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("program.id"), nullable=True
    )
    parent: Mapped[Optional["ProgramModel"]] = relationship(
        "ProgramModel",
        remote_side=[id],
        back_populates="children",
    )
    children: Mapped[list["ProgramModel"]] = relationship(
        "ProgramModel",
        back_populates="parent",
    )
    inspired_by: Mapped[list["ProgramModel"]] = relationship(
        "ProgramModel",
        secondary=inspiration,
        primaryjoin=id == inspiration.c.inspired_by_id,
        secondaryjoin=id == inspiration.c.inspired_id,
        backref="inspired",
    )
```

### Island-based parallel evolution
This is what sets AlphaEvolve apart from previous work.

```python
def run_pipeline(
    parent: Program,
    inspirations: list[Program],
    cfg: PipelineConfig,
    api_key: str,
) -> Program | Exception:
    try:
        prompt = build_prompt(parent, inspirations, cfg.prompt_builder_cfg)
        diff, reasoning = generate_diff_and_reasoning(prompt, api_key, cfg.llm_cfg)
        child_content = apply_diff(parent.content, diff)
        scores, artifacts = evaluate_program(child_content, cfg.program_evaluator_cfg)
        child = Program(
            id=None,
            island_id=parent.island_id,
            generation=parent.generation + 1,
            content=child_content,
            parent_id=parent.id,
            inspired_by_ids=[insp.id for insp in inspirations if insp.id is not None],
            prompt=prompt,
            reasoning=reasoning,
            diff=diff,
            scores=scores,
            artifacts=artifacts,
        )
        return child
    except Exception as e:
        return e
```
```python
@ray.remote
def run_pipeline_task(
    parent: Program, inspirations: list[Program], cfg: PipelineConfig, api_key: str
) -> Program | Exception:
    return run_pipeline(parent, inspirations, cfg, api_key)
```

```python
while True:
    self._process_task_completions()
    done = self._start_new_tasks()
    if done:
        break
    self._migrate()
    time.sleep(0.1)  # Sleep to avoid busy-waiting

logger.info(f"Waiting for {self.n_running_tasks} running tasks to finish...")
self._wait_for_all_tasks()
```

```python
for island_id, island in enumerate(self.islands):
    if len(self.island_tasks[island_id]) >= island.cfg.max_parallel_tasks:
        # Skip this island if it has reached the max number of running tasks
        continue

    parent, inspirations = island.sample()
    task = run_pipeline_task.remote(
        parent,
        inspirations,
        self.cfg.pipeline_cfg,
        self.api_key,
    )
    task_id = id(task)
    self.island_tasks[island_id][task_id] = task
    logger.info(f"Started task {task_id} on island {island_id}.")
```

## Results
### Circle packing

