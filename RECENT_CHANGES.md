# Recent Improvements

This file records improvements made after for the ASE Tool Track submission.

## Framework and Parsing Support

- Added support for Codex and Claude Code session logs alongside SWE-agent,
  mini-swe-agent, and OpenHands inputs.
- Expanded shell and tool parsing for wrapped commands, editor actions,
  planning actions, final answers, and framework-specific metadata.
- Added the `plan` phase for explicit planning operations in Codex and Claude
  Code trajectories.
- Improved context-sensitive classification of validation commands and action
  success/failure indicators.

## Viewer Improvements

- Added background trajectory loading and graph prefetching so navigation stays
  responsive while larger collections are indexed.
- Added file-footprint exploration, node-linked file highlighting, settings,
  help/legend controls, and improved trajectory-title handling.
- Added support for observation indicators, compressed `cd` nodes, unique think
  nodes, and clearer phase/edge explanations in the tutorial.
- Improved graph rendering for long thoughts, arrow scaling, phase colors, and
  interactive controls.
