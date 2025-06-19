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

This didn't sit right with me, as I subscribe to the view that advanced AIs shouldn't need human hand-holding to achieve results, à la David Silver's and Richard Sutton's ["Era of Experience"](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf) thesis (I highly recommend this [podcast episode](https://youtu.be/zzXyPGEtseI?si=qW4ek0_13GIvt_6M) on the topic). So, I looked into what it would take to Do It Myself.

{% include image.html url="/assets/images/alpha-evolve/vs_funsearch.png" description="FunSearch vs AlphaEvolve" %}

I referenced the above table from the AlphaEvolve paper to see what the main engineering challenges would be, and summed them up as follows:

1. Storing more data to enrich prompt content
2. Using diff-based evolutionary steps to support more complex programs
3. Exploiting parallelism to increase throughput while maintaining program diversity

There were lots of aspects to this project that excited me: I like maths, and I'd recently dabbled in an agentic coding project involving code diffs, parallelised processing, and databases for LLM training. I figured, what the hell, let's slap my name on it and try to make my own replication with cleaner prompting. Thus was born [AlfredEvolve](https://github.com/alfredclwong/alfred-evolve).

## Replication
### Pipeline
As seen in Fig 1., AlphaEvolve consists of an evolution loop that iterates through the following steps:

1. Sample a parent program and inspiration programs from the program database
2. Build a prompt using the sampled programs
3. Generate a diff by passing the prompt to an LLM
4. Apply the diff to the parent to get a child program
5. Evaluate the child program to get scores
6. Store the results

Steps 1 & 6 involve database operations, which need careful handling in parallelised pipelines, so I separated them from the core steps 2-5 for now.

```python
# Steps 2-5 of the AlphaEvolve pipeline
prompt = build_prompt(parent, inspirations, cfg.prompt_builder_cfg)
diff, reasoning = generate_diff_and_reasoning(prompt, api_key, cfg.llm_cfg)
child_content = apply_diff(parent.content, diff)
scores, artifacts = evaluate_program(child_content, cfg.program_evaluator_cfg)
```



### Data model
In some sense, AlphaEvolve is a very simple 6-step algorithm. However, as I've learnt over the course of several coding projects, simple things quickly spiral out of control if you don't keep the data structures tidy, especially so when parallelism is involved.

The central data structure is the Program, with a few key concepts:

- Each Program belongs to an evolutionary tree of Programs
- Diff-based evolution means that each non-root Program has a parent Program and diff string
- The Program tree is divided into islands
- 

```python
@dataclass
class Program:
    id: Optional[int]
    island_id: int
    generation: int
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


### Parallelism
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

### Island-based evolution
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

