# Baseline results — checkpoint

Mean ± std over 5 seed(s) (42–46). Tree model: `XGBoost`. Threshold for macro-F1 tuned on validation.

## Main table

| run                                      | split         | AUC-ROC       | AP            | macro-F1      | P@top1%       | tests | seeds |
|------------------------------------------|---------------|---------------|---------------|---------------|---------------|-------|-------|
| E1a  text: TF-IDF word+char + LR         | user_disjoint | 0.753 ± 0.004 | 0.344 ± 0.005 | 0.636 ± 0.004 | 0.612 ± 0.044 | H1    | 5     |
| E2   behavior + LR                       | user_disjoint | 0.786 ± 0.006 | 0.316 ± 0.011 | 0.626 ± 0.007 | 0.463 ± 0.026 | H1    | 5     |
| E2   behavior + boosting                 | user_disjoint | 0.793 ± 0.006 | 0.359 ± 0.013 | 0.629 ± 0.007 | 0.680 ± 0.059 | H1    | 5     |
| E4   behavior + text stats + text score  | user_disjoint | 0.838 ± 0.002 | 0.442 ± 0.012 | 0.679 ± 0.005 | 0.758 ± 0.036 | H2    | 5     |
| Leak-1  behavior + useful/funny/cool     | user_disjoint | 0.891 ± 0.003 | 0.529 ± 0.012 | 0.713 ± 0.005 | 0.861 ± 0.035 | leak  | 5     |
| Leak-2  behavior, RANDOM split           | random        | 0.806 ± 0.002 | 0.364 ± 0.003 | 0.636 ± 0.002 | 0.680 ± 0.011 | H3    | 5     |
| E2-time  behavior, time split            | time          | 0.730 ± 0.003 | 0.264 ± 0.005 | 0.581 ± 0.004 | 0.474 ± 0.045 | H1    | 5     |
| E2-domain  behavior, restaurants->hotels | domain        | 0.593 ± 0.007 | 0.209 ± 0.003 | 0.487 ± 0.008 | 0.200 ± 0.016 | H1    | 5     |

## By reviewer history (H4 baseline: one-review accounts are the hard case)

| run                                      | AUC one-review | AUC repeat    | gap    |
|------------------------------------------|----------------|---------------|--------|
| E1a  text: TF-IDF word+char + LR         | 0.710 ± 0.006  | 0.764 ± 0.011 | +0.054 |
| E2   behavior + LR                       | 0.614 ± 0.008  | 0.813 ± 0.015 | +0.199 |
| E2   behavior + boosting                 | 0.642 ± 0.007  | 0.805 ± 0.014 | +0.163 |
| E4   behavior + text stats + text score  | 0.725 ± 0.008  | 0.855 ± 0.012 | +0.129 |
| Leak-1  behavior + useful/funny/cool     | 0.815 ± 0.003  | 0.903 ± 0.007 | +0.087 |
| Leak-2  behavior, RANDOM split           | 0.634 ± 0.002  | 0.853 ± 0.004 | +0.220 |
| E2-time  behavior, time split            | 0.577 ± 0.004  | 0.769 ± 0.005 | +0.193 |
| E2-domain  behavior, restaurants->hotels | 0.562 ± 0.007  | 0.649 ± 0.006 | +0.087 |

## Hypotheses

- **H1** (behavior beats text): AUC 0.793 vs 0.753 (+0.040); AP 0.359 vs 0.344 (+0.015) -> supported on AUC.
- **H2** (combining beats the best single signal in AP): E4 AP 0.442 vs best single 0.359 (+0.083) -> supported.
- **H3** (random split inflates AUC): random 0.806 vs user-disjoint 0.793 (+0.012) -> supported.
- **H4 motivation** (where each signal works): on one-review accounts text scores 0.710 against behavior's 0.642, while on repeat reviewers behavior scores 0.805 against text's 0.764. The two signals are strong on **opposite** groups, which is the case for keeping ego features separate and for fusing signals rather than picking one.
- **RQ4** (restaurants → hotels): AUC 0.593 against 0.793 in-domain, i.e. well above chance but far below in-domain — behavior features transfer only weakly. Hotel-only reviewers have a different activity profile from the restaurant reviewers the model was fitted on. This narrows the scope-of-claims slide: transfer degrades badly even *within* Yelp. Note the verdict is model-dependent — HistGradientBoosting scored 0.51 here where XGBoost scores 0.59, so report the model alongside the number.
- **Vote leak**: adding useful/funny/cool moves AUC 0.793 -> 0.891 (+0.098) and AP 0.359 -> 0.529. Confirms the columns must stay out.

## Caveats to state on the slide

- Behavior features are computed once over the **whole** dataset (from `date, user_id, product_id, star` only, never labels). For the user-disjoint, random and domain splits this is the transductive setting the task definition declares. For **E2-time it is a temporal leak**: a 2010 training review's `u_n` and `u_active_days` already count its reviewer's 2012 activity, so that row is optimistic and is reported as a robustness check only.
- `Leak-1` and `Leak-2` are diagnostics, not candidate models. They measure what the vote columns and the random split would buy; neither is used anywhere else.
- Macro-F1 depends on a threshold tuned on validation, so it moves more across seeds than AUC or AP. AP is the headline metric for a 13.2% positive rate.
