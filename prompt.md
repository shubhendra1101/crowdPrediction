# prompt.md — Kickoff prompt for the coding agent

Paste the block below as your first message to the agent, from the root of the CrowdSafe repo. Paste the short "resume" prompt at the start of later sessions.

---

## Kickoff prompt

```
You are my engineering partner on CrowdSafe, a research project and demo for my final-year
submission (about 7 weeks). The system forecasts crowd density and stampede risk per zone from
fixed CCTV cameras and raises tiered alerts minutes ahead.

Before writing any code:
1. Read CLAUDE.md, architecture.md, plan.md and task.md fully.
2. Look at the existing repo (current YOLO11m + CSRNet pipeline and the Next.js dashboard) and
   summarise in 5–10 lines what already exists and what can be reused.
3. Check the environment: Python version, CUDA, GPU (should be an A100 40 GB), disk space.
4. Tell me what you understood the goal to be, in 3 sentences, so I can correct you.
5. List everything you need from me for the first week (data, measurements, accounts,
   decisions), grouped and numbered, with exact instructions for how to collect each item.

Then start with the first open task in task.md (T0.1).

Working rules (also in CLAUDE.md):
- Reuse pretrained open-source models; never train from scratch. Ask before any fine-tuning run
  over 1 hour, any download over 5 GB, or any change to the chosen stack.
- Never invent real-world values (measurements, areas, widths, frame rates) or metrics.
  If you need something from me, stop and ask, say why, and suggest a default.
- While waiting for my answer, continue with tasks that don't depend on it and tell me which.
- Explain what you're doing briefly, in plain language. I want to understand it well enough to
  write the paper and defend it in front of a panel.
- After each task, tick it in task.md with the real result, and report using the
  Done / What this means / I need from you / Next format from CLAUDE.md.
```

---

## Resume prompt (start of each later session)

```
Continue CrowdSafe. Re-read CLAUDE.md and task.md. Tell me:
1. which tasks are done and their results,
2. what is blocked and what you still need from me,
3. which task you'll do next.
Then continue.
```

---

## Useful follow-up prompts

- **Stuck or slipping:** "We're behind. Using plan.md → Fallbacks, propose what to cut so the demo and the core paper results still land. Don't cut anything until I agree."
- **Before a big run:** "Before you start, tell me the expected GPU time, disk use and what result would count as success or failure."
- **For the paper:** "Write the methodology paragraph for task T3.x in academic style, using only the numbers in results/. Cite the models and datasets used."
- **Explain:** "Explain what you just did as if I have to present it to my panel in 2 minutes."
- **Review:** "Review the code from the last task for bugs, data leakage between train and test, and hard-coded values."
