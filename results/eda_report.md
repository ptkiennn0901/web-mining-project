# EDA — YelpChi

`67,395` reviews · `8,919` fake (13.2%) · `38,063` reviewers · `201` products · Oct 2004 – Oct 2012

## 1. Dataset shape

| domain     | reviews | fake  | fake rate | products |
|------------|---------|-------|-----------|----------|
| hotel      | 5,854   | 778   | 13.3%     | 72       |
| restaurant | 61,541  | 8,141 | 13.2%     | 129      |

## 2. Fake vs genuine

| feature                                 | fake  | genuine |
|-----------------------------------------|-------|---------|
| Reviews with any useful/funny/cool vote | 0.0%  | 49.9%   |
| Median length (words)                   | 72    | 112     |
| Mean length (words)                     | 102   | 144     |
| Extreme rating (1 or 5 stars)           | 56.7% | 40.8%   |
| 1-star share                            | 12.8% | 4.0%    |
| 5-star share                            | 43.8% | 36.8%   |
| Mean abs. deviation from product mean   | 1.07  | 0.80    |
| Mean reviews by the same reviewer       | 1.46  | 4.70    |
| Mean star                               | 3.80  | 3.96    |

## 3. Four findings that shape the design

| finding                                  | number                                                                                                   | design decision                                        |
|------------------------------------------|----------------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| Votes leak the label                     | 0 of 8,919 fakes carry a vote (genuine: 49.9%)                                                           | drop useful/funny/cool                                 |
| Labels are nearly per-reviewer           | 135 of 38,063 reviewers have both labels; same-reviewer review pairs share a label 99.7% of the time     | main split is user-disjoint                            |
| Most fakes come from one-review accounts | 70.6% of reviewers are singletons; they are 39.8% of reviews and 76.0% of fakes; fake rate 25.3% vs 5.3% | model must decide from a review's own features         |
| Product relations are weak               | a fake's same-product-month neighbors are only 16.6% fake (same-star: 18.8%)                             | filter product neighbors, separate weight per relation |

## 4. Review graph

| relation | edges     | label homophily |
|----------|-----------|-----------------|
| RUR      | 110,112   | 0.997           |
| RTR      | 466,352   | 0.792           |
| RSR      | 8,166,840 | 0.804           |

## 5. Fake rate by reviewer activity

| reviews by reviewer | reviews  | fake  | fake rate | share of all fakes |
|---------------------|----------|-------|-----------|--------------------|
| 1                   | 26,855.0 | 6,781 | 25.3%     | 76.0%              |
| 2                   | 11,622.0 | 1,332 | 11.5%     | 14.9%              |
| 3                   | 6,411.0  | 413   | 6.4%      | 4.6%               |
| 4-5                 | 7,483.0  | 229   | 3.1%      | 2.6%               |
| 6-10                | 8,139.0  | 130   | 1.6%      | 1.5%               |
| 11+                 | 6,885.0  | 34    | 0.5%      | 0.4%               |

## Figures

- `figures/fig1_star_by_label.png`
- `figures/fig2_length_by_label.png`
- `figures/fig3_reviews_per_reviewer.png`
