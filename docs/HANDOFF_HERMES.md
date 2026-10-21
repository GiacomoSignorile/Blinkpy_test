# Handoff prompt for Hermes

Continue the work on the repo /home/giacomo/Blinkpy_test, branch `research/pseudolabel-improvements` (work only here, commit locally, never touch `main`, never switch the production model in docker-compose.yml). Read docs/research_notes.md first: it holds every result so far.

Context: Blink camera project that identifies household members from person crops (9 classes, 569 human-labelled crops, 1325 unlabelled). Only HUMAN labels may be used for evaluation (clip-grouped and temporal folds); do not label anything by hand.

GPU server: `ssh giacomo@100.112.122.139` (RTX 5070 12 GB), work dir ~/blinkwatch-train, venv .venv, data in export5 (+ export5_manifest.csv). Copy scripts there with scp from training/. Do NOT use `pkill -f` / `pgrep -f` with a pattern that matches your own ssh command line (use 'python name[.]py'). Launch long jobs with `(nohup ... > log 2>&1 < /dev/null &)` and wait with an until-loop on the .done file.

Findings so far:
- Pseudo-labels: plain logistic regression with a global threshold beats label spreading, per-class thresholds and kNN reject (DINOv2-L+CLIP-L: 95.5% precision at 39% coverage).
- CLIP-L fine-tuning recipes (LP-FT, LLRD, strong aug, WiSE-FT) do not beat the current recipe; differences are within fold noise.
- Frozen backbones (balanced acc., group/time): PE core L 75.3/78.9 (best), EVA02-CLIP L 68.1/62.8, CLIP-L 65.0/65.9, DINOv3 L 64.7/58.6, SigLIP2 so400m 64.4/61.5, DINOv2-L 56.1/57.2. Concatenating backbones does not help.

Running now on the server (started ~19:40): `ft_backbones.py --arch vit_pe_core_large_patch14_336.fb --seeds 3 --epochs 10`, log ft_pe.log, done-marker ft_vit_pe_core_large_patch14_336.done, results results_ft_vit_pe_core_large_patch14_336.json. It compares single seed / ensemble / model soup per fold (3 group + 3 time folds, ~2 h total).

To do:
1. When it finishes, append the tables to docs/research_notes.md and commit. Compare with CLIP-L base (group 76.9, time 69.2, same folds) and v3 (75.9 / 78.1, different folds).
2. If PE fine-tuned/soup beats CLIP-L on both protocols, try on the unlabelled pool: pseudo-label with PE frozen features (autolabel_v2.py --combo needs feat_*.npy; backbone_bench.py caches feat_bb_pe_core_L_569.npy for human crops only, extend it to all 1894 persona rows) and check precision/coverage against the CLIP-L numbers.
3. If PE wins, port it to the watcher: blinkwatch/person_model.py loads a checkpoint with arch/arch_kwargs/mean/std (see training/train_final.py for the format); add PE as an option via train_final.py, do not change the default.
4. Update the unit tests in tests/ if you touch blinkwatch/, run them with .venv/bin/python -m pytest.
5. Report to the user in Italian, honestly: say when differences are inside the fold-to-fold noise.

The user needs to run the push themselves: `git push -u origin research/pseudolabel-improvements` (the permission system blocked it for Claude).
