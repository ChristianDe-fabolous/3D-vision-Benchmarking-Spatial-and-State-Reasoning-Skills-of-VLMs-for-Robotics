# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Hard Rules

- **Never `git commit` or `git push`** under any circumstances.
- **Never modify anything in `project-proposal/`** — all LaTeX files are off-limits.
- **All code work happens in `src/`** — this is the primary working directory for implementation.

## Project Overview

This is a 3D Vision course project (ETH Zurich, SS26) benchmarking Vision-Language Models (VLMs) for spatial and state reasoning in robotics. Two research directions:

1. **Robotic failure mode detection**: Can VLMs correctly identify the current state/failure mode of a robotic task from images?
2. **Multi-view consistency**: Can VLMs extract spatial information from multiple camera angles, and does providing multiple perspectives improve accuracy?

Primary model under evaluation: Qwen (potentially others like Pi, RT2).

## Dataset Notes

- Dataset entries with known wrong answers: #5, #35, #37, #67, #69
- Failure mode task: images may be presented in reverse order (as if robot did the opposite action)
- Multi-view task: two camera angles available; one with spatial reference points


## Note
Afterwards your code will be checked by Codex
