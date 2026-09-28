# MLB Hitting vs. Pitching: Predicting Winning Seasons

## Research Question

**Is good pitching or good hitting more likely to lead a team to wins in the MLB?**

For the machine-learning analysis, this is operationalized as: **Can an MLB team's hitting statistics or pitching/run-prevention statistics better predict whether the team finishes a season with a winning record?**

This is a binary classification project. A team-season is labeled as a winning season when W > L.

## Dataset

- Source: Lahman Baseball Database Teams table
- Repository: https://github.com/corbtastik/lahman-baseball-db
- Years: 2000–2025
- Observations: 780 team-season records
- Unit: one MLB team in one season
- Target: Winning_Season (1 = W > L; 0 = W <= L)
- Class balance: 392 winning / 388 non-winning
- Training: 2000–2022 (690)
- Testing: 2023–2025 (90)

Using a winning-record target instead of raw wins makes the target more comparable across seasons, including the shortened 2020 season.

## Features

### Hitting
- Runs per game
- Home runs per game
- On-base percentage (OBP)
- Slugging percentage (SLG)

### Pitching / Run Prevention
- Runs allowed per game
- ERA
- Strikeouts per 9 innings

Wins, losses, winning percentage, and other direct outcome variables are excluded from predictors to prevent target leakage.

## Data Preparation

Python/pandas are used for filtering and feature engineering. The selected raw variables required for the features have 0 missing values, and there are 0 duplicate team-season keys. The selected features are numeric, so categorical encoding is unnecessary. Logistic Regression uses StandardScaler inside a pipeline. The Decision Tree is limited to max_depth=4 for interpretability and to reduce unnecessary complexity.

## Exploratory Analysis

The notebook includes summary statistics, class balance, feature relationships, correlations, and IQR-based outlier checks. The classes are nearly perfectly balanced, so there is no severe class-imbalance problem.

## Training and Testing

A chronological split is used:

- Training: 2000–2022
- Testing: 2023–2025

The test seasons remain unseen during model training. Because the predictors and target come from the same season, this is **contemporaneous predictive classification**, not a pre-season forecast.

## Baseline

A majority-class DummyClassifier is used as the baseline. The 2023–2025 test baseline accuracy is approximately **54.4%**.

## Models

1. Logistic Regression
2. Decision Tree Classifier (max_depth=4)

Each model is trained using hitting-only, pitching-only, and combined feature sets.

## Results

| Model | Feature Set | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Logistic Regression | Hitting | 73.3% | 83.8% | 63.3% | 72.1% |
| Decision Tree | Hitting | 73.3% | 93.1% | 55.1% | 69.2% |
| Logistic Regression | Pitching | 84.4% | 81.8% | 91.8% | 86.5% |
| Decision Tree | Pitching | 76.7% | 72.6% | 91.8% | 81.1% |
| Logistic Regression | Combined | 87.8% | 91.3% | 85.7% | 88.4% |
| Decision Tree | Combined | 86.7% | 87.8% | 87.8% | 87.8% |

### Interpretation

When considered separately, the pitching/run-prevention group produced stronger predictive performance than the selected hitting group.

- Best hitting-only accuracy: 73.3%
- Best pitching-only accuracy: 84.4%
- Best combined accuracy: 87.8%

The combined Logistic Regression model correctly classified 79 of 90 held-out team-seasons.

These results describe predictive performance, not proof that pitching causes winning seasons.

## Confusion Matrix

For the combined Logistic Regression model:

| | Actual Non-Winning | Actual Winning |
|---|---:|---:|
| Predicted Non-Winning | 37 | 7 |
| Predicted Winning | 4 | 42 |

There were 4 false positives and 7 false negatives.

## Model Interpretation

The notebook reports Logistic Regression coefficients, Decision Tree feature importance, confusion matrices, and incorrect predictions. These outputs explain model behavior rather than establish causal effects.

## Ethics and Limitations

- Observational team-season data cannot establish causation.
- Same-season predictors and outcome mean this is not a pre-season forecasting model.
- Defense, baserunning, injuries, roster construction, payroll, park effects, schedule strength, and bullpen usage are omitted.
- The test period contains only three seasons.
- The 2020 season was shortened to 60 games.
- MLB rules, technology, strategy, and offensive environments changed across 2000–2025.
- A real-world model used for betting, personnel, financial, or operational decisions could have greater consequences when predictions are wrong.

## Supporting Research

1. Baseball-Reference. (n.d.). *Major League Baseball statistics and history*. https://www.baseball-reference.com/
2. FanGraphs. (n.d.). *FanGraphs Library Glossary*. https://library.fangraphs.com/fangraphs-library-glossary/
3. Albert, J., & Bennett, J. (2001). *Curve Ball: The Complete Guide to the Science of Baseball*. Copernicus.
4. Major League Baseball. (n.d.). *On-base percentage*. MLB Glossary. https://www.mlb.com/glossary/standard-stats/on-base-percentage
5. Major League Baseball. (n.d.). *Slugging percentage*. MLB Glossary. https://www.mlb.com/glossary/standard-stats/slugging-percentage
6. Major League Baseball. (n.d.). *Earned run average*. MLB Glossary. https://www.mlb.com/glossary/standard-stats/earned-run-average

## AI / Code Transparency

OpenAI ChatGPT (GPT-5.6 Luna) was used to assist with project planning, research-question refinement, code structure, model setup, interpretation, and written explanations. The student is responsible for reviewing and understanding the submitted code and following course AI-use requirements.

## Rubric Coverage

The repository and notebook cover the ML problem, dataset, background research, preparation, exploratory analysis, train/test strategy, baseline, two models, evaluation metrics, model interpretation, ethics/limitations, and AI/code transparency.

## Portfolio

Project 02: https://jasonhowe2.github.io/Personal-Portfolio-Project/project2.html
