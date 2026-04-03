# Week 1 Analysis: CodeIt Pipeline Review & Human Bias Insertion Points

## 1. Environment Setup

### Dependencies
- Python 3.11 compatible (required relaxing `numpy<=1.23` to `numpy<2`)
- Key packages: `torch`, `lightning==2.0.2`, `transformers>=4.35`, `datasets==2.9.0`, `peft==0.10.0`
- Model: `codet5-small` (smallest), `codet5p-220m` (default), `codet5p-770m` (largest)
- Need GPU for training — use NYU HPC (Greene cluster)

### Path Fix
`codeit/__init__.py` line 7: `PROJECT_FOLDER_PATH` uses `split("codeit")[0]` which breaks when the repo folder is also named `codeit`. Fixed to use `os.path.dirname()`.

### Data Pipeline Verified
- 401 training tasks, 401 evaluation tasks
- Each task has: `program`, `training_examples`, `test_examples`, `task_key`, `extra_info`
- 160 DSL primitives available
- Ground truth programs execute correctly on training examples
- Mutated task files: 8,427 tasks in `mutated_tasks_train_9600.json`
- Split: 311 train / 89 val / 400 test

---

## 2. Pipeline Architecture

```
Meta-iteration loop (99 iterations default):
  1. Sample replay buffer → training data
  2. Train model (CodeT5+) on sampled data
  3. Sample programs from trained policy
  4. Evaluate programs on ARC tasks
  5. Add successful programs to replay buffer (hindsight relabeling)
  6. Update priorities in buffer
  7. Log metrics
```

---

## 3. Feasible Insertion Points for Human Bias

### Priority Level: HIGH (most impactful, easiest to implement)

#### A. Replay Buffer Priority Function (`codeit/replay_buffer.py`, `get_priority()`)
- **Current formula**: weighted sum of program_length, age, distance, task_demonstration_performance
- **Modification**: Add `human_solution_similarity * human_bias_penalty` term
- **Why**: Directly controls which programs the model learns from. Programs similar to human H-ARC traces get higher priority → model trains on more human-like solutions.
- **Implementation effort**: Low — add one new term to existing priority formula

#### B. Reward Function (`codeit/policy/inference.py`, `evaluate_actions()`)
- **Current**: `reward = 1 - abs_grid_distance(output, target)`
- **Modification**: Add human similarity bonus: `reward += human_bonus * WEIGHT`
- **Why**: Shapes what the policy considers "good" — not just correct output, but human-like process
- **Implementation effort**: Low — modify reward computation

#### C. Training Loss Weighting (`codeit/hf_model_module.py`, `training_step()`)
- **Current**: Standard cross-entropy loss, equal weight for all examples
- **Modification**: Weight loss per example by human-solution similarity
- **Why**: Model learns more from human-like examples
- **Implementation effort**: Low — multiply loss by per-example weight

### Priority Level: MEDIUM

#### D. Mutation Bias (`codeit/augment/genetic.py`)
- **Current**: Random mutations with fixed probabilities (phi_program, phi_var, phi_func, phi_arg)
- **Modification**: Learn mutation probabilities from H-ARC solution patterns
- **Why**: Generate mutated tasks that are more human-like in structure
- **Implementation effort**: Medium — need to analyze H-ARC data to extract mutation statistics

#### E. Sampling Temperature / Logits Processor (`codeit/policy/inference.py`, `generate()`)
- **Current**: `temperature=0.95`, only `BadTokenEnforcedEndLogitsProcessor`
- **Modification**: Add `HumanBehaviorLogitsProcessor` that biases token probabilities toward human-preferred DSL primitives
- **Why**: During program generation, favor primitives that humans use more often
- **Implementation effort**: Medium — need to build primitive frequency distribution from H-ARC

### Priority Level: LOW (experimental, harder to implement)

#### F. Experience Ratio (`codeit/exit_data_module.py`)
- **Current**: 90,000 policy + 10,000 mutated experiences per iteration
- **Modification**: Add third category "human" experiences, or adjust ratio dynamically
- **Implementation effort**: Medium-High

#### G. Inference Dataset Augmentation (`codeit/agent.py`)
- **Current**: Standard ARC tasks
- **Modification**: Include H-ARC-derived tasks with human behavioral annotations
- **Implementation effort**: High — need H-ARC data preprocessing pipeline

---

## 4. Minimum Baseline Outputs to Save

For fair comparison between baseline and human-biased models, save these at each meta-iteration:

### Task-Level Metrics (per iteration)
1. `cumulative_performance` — fraction of evaluation tasks solved (test set)
2. `task_demonstration_performance` — fraction of training examples solved
3. `num_policy_tasks` — number of unique tasks in policy buffer
4. `num_mutated_tasks` — number of unique tasks in mutated buffer
5. `num_policy_programs` — total programs in policy buffer

### Trace-Level / Process Metrics (per iteration)
6. **Sampled programs** — full program text for all generated programs (`log_{i}.json`)
7. **Program length distribution** — number of lines per program
8. **DSL primitive usage** — frequency of each primitive across all sampled programs
9. **Solution diversity** — number of distinct programs per task

### Human-Likeness Metrics (for later comparison with H-ARC)
10. **Intermediate states** — grid state after each program line execution
11. **Subgoal structure** — which DSL operations are used as "chunks"
12. **Program complexity** — AST depth, number of variables, function composition depth
13. **Solution trajectory** — sequence of grid transformations from input to output

### Files to Save Per Run
- `performance.csv` — already exists, add columns for items 6-9
- `log_{i}.json` — already exists, contains sampled programs
- `solutions_{i}.json` — already exists, contains solutions
- **NEW**: `trace_{i}.json` — intermediate states and DSL usage per program
- **NEW**: `program_stats_{i}.json` — program length, complexity, primitive frequency

---

## 5. Recommended Implementation Order

1. **Week 2**: Add process-level logging (items 6-13 above)
2. **Week 3**: Implement insertion point A (replay buffer priority) + B (reward modification)
   - These are the highest-impact, lowest-effort changes
   - Need: H-ARC data loader, human similarity metric
3. **Week 4**: Run experiments — baseline vs. A-only vs. B-only vs. A+B
4. **Week 5**: Evaluate using human-likeness metrics from Week 2 logging

---

## 6. Key Files Reference

| File | Role | Lines of Interest |
|------|------|-------------------|
| `run/run_codeit.py` | Main loop | L189-287: meta-iteration loop |
| `codeit/agent.py` | Agent orchestration | L204-277: policy sampling |
| `codeit/replay_buffer.py` | Experience storage & priority | L214-251: `get_priority()` |
| `codeit/policy/inference.py` | Program generation & evaluation | L92-119: `generate()`, L126-140: `evaluate_actions()` |
| `codeit/augment/genetic.py` | Program mutation | L228-336: `mutate_task()` |
| `codeit/exit_data_module.py` | Training data sampling | L138-148: `setup()` |
| `codeit/hf_model_module.py` | Model training | L75-80: `training_step()`, L127-131: `step()` |
| `run/config/base_config.yaml` | Configuration | All default parameters |
