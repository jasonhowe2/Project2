# MLB Hitting vs. Pitching: Predicting Team Wins

## Research Question
**How well can MLB team wins be predicted using hitting statistics compared with pitching/run-prevention statistics?**

## Project Overview
This machine-learning project uses MLB team-season data from 2000–2025 to predict team wins. The target variable is **W (wins)**, making this a **regression** problem.

The project compares:
- Hitting-only models
- Pitching/run-prevention-only models
- Combined hitting + pitching models
- Linear Regression
- K-Nearest Neighbors Regression

## Dataset
- **Source:** Lahman Baseball Database Teams table
- **Source repository:** https://github.com/corbtastik/lahman-baseball-db
- **Unit of analysis:** One MLB team in one season
- **Years:** 2000–2025
- **Observations:** 780 team-season records
- **Target:** W (team wins)

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

## Method
The project uses a chronological train/test split:
- **Training:** 2000–2022 (690 team-season records)
- **Testing:** 2023–2025 (90 team-season records)

A chronological split was selected so the final seasons are genuinely unseen during model training and to reduce temporal leakage. The model uses season-level statistics from the same season as the win total, so this should be interpreted as **contemporaneous predictive modeling**, not a pre-season forecast.

KNN uses standardization because it is distance-based. The project establishes a mean-wins baseline and compares models using MAE, RMSE, and R².

### Data-quality checks
The selected Lahman records for 2000–2025 contain **0 missing values** in the raw variables needed for the engineered features and **0 duplicate team-season keys**. Because this is a regression problem, class imbalance is not applicable. Summary statistics and scatterplots are used to investigate unusual distributions and potential outliers. The 2020 season is specifically noted as a limitation because MLB played a shortened schedule.

## Baseline Performance
The baseline predicts the mean training-set win total (**78.75 wins**) for every team in the 2023–2025 test set. On the held-out test set, the baseline has **MAE = 9.77, RMSE = 12.44, and R² = -0.033**. The negative R² is expected for a baseline that is weaker than the variation in the test outcomes. The machine-learning models are therefore compared against this simple reference.

## Results
Held-out test results:

| Model | Feature Set | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Linear Regression | Hitting | 9.02 | 10.57 | 0.254 |
| Linear Regression | Pitching | 8.11 | 10.27 | 0.295 |
| Linear Regression | Combined | 7.92 | 9.01 | 0.458 |
| KNN (k=7) | Hitting | 8.28 | 10.27 | 0.295 |
| KNN (k=7) | Pitching | 7.85 | 10.10 | 0.318 |
| KNN (k=7) | Combined | 6.32 | 8.12 | 0.559 |

The combined KNN model produced the strongest held-out performance among the tested models. Pitching/run prevention performed somewhat better than the selected hitting variables when the groups were considered separately.

## Interpretation
The results suggest that both run creation and run prevention provide useful predictive information about team wins. The combined KNN model improves substantially on the mean-wins baseline, but the project does **not** establish that pitching or hitting causes wins; it evaluates predictive performance using the selected statistics and seasons.

Because this is regression, the project does not use the terms **false positive** and **false negative**. Instead, prediction errors are described as **overpredictions** (predicted wins are too high) or **underpredictions** (predicted wins are too low). Large errors could matter if a similar model were used for roster, financial, operational, or betting decisions.

## Limitations
The model does not include every factor that can affect wins, including defense, baserunning, injuries, roster construction, payroll, strength of schedule, park effects, and bullpen usage. The test period contains only three seasons, so performance may differ on another time period. The 2020 shortened season also makes raw win totals less directly comparable across the full study period. Finally, because the predictors and target come from the same season, the model should not be presented as a pre-season forecasting system.

## Repository Files
- `Project2_MLB_Hitting_vs_Pitching.ipynb` — complete end-to-end Jupyter Notebook
- `project2_mlb_hitting_vs_pitching.py` — supporting Python analysis code
- `README.md` — project overview and documentation

## Background Sources
1. Baseball-Reference. (n.d.). *Major League Baseball statistics and history*. https://www.baseball-reference.com/
2. FanGraphs. (n.d.). *FanGraphs library glossary*. https://library.fangraphs.com/fangraphs-library-glossary/
3. Albert, J., & Bennett, J. (2001). *Curve ball: The complete guide to the science of baseball*. Copernicus.

## AI / Code Transparency
OpenAI ChatGPT (GPT-5.6 Luna) was used to assist with project planning, research-question refinement, code structure, model setup, interpretation, and written explanations. The student is responsible for reviewing and understanding the submitted code and results and for following the course's AI-use requirements.

## Assignment Coverage
The repository includes the major Project 2 requirements: problem definition, background/context, data description, exploratory analysis, feature selection, preprocessing, chronological train/test strategy, baseline, multiple models, model comparison, evaluation metrics, model interpretation/error analysis, limitations/ethics, and AI/code transparency.


## Rubric Coverage

| Requirement | Where it is addressed |
|---|---|
| 1. Published Portfolio Project | Project 02 is published on the personal portfolio as `project2.html`. |
| 2. ML Problem & Dataset | Notebook sections 1–3 and the Project 02 portfolio page define the regression target, unit of analysis, features, dataset, and sample size. |
| 3. Context & Supporting Research | Notebook/background section and portfolio page include three cited credible sources. |
| 4. Data Preparation | Feature engineering, missing-value handling, rate transformations, and KNN scaling are documented in the notebook. |
| 5. Data Understanding & Feature Selection | Summary statistics, scatterplots, correlations, and feature-group rationale are included. |
| 6. Training & Testing Strategy | Chronological 2000–2022 training / 2023–2025 testing split is documented. |
| 7. Baseline Performance | Mean-training-wins baseline is calculated and evaluated with MAE, RMSE, and R². |
| 8. Model Development & Comparison | Linear Regression and KNN Regression are compared across three feature sets. |
| 9. Model Evaluation | MAE, RMSE, and R² are defined, reported, and used for comparison. |
| 10. Model Interpretation | Actual-vs-predicted analysis, linear coefficients, and largest-error analysis are included. |
| 11. Ethics & Limitations | Limitations, uncertainty, omitted variables, consequences of errors, and real-world use are discussed. |
| 12. Code & AI Transparency | Repository, notebook, data source, and AI-use disclosure are documented. |

## Portfolio

**Project 02 portfolio page:** https://jasonhowe2.github.io/Personal-Portfolio-Project/project2.html

**Portfolio homepage:** https://jasonhowe2.github.io/Personal-Portfolio-Project/

## Distinction-Focused Work

- Clear project-page hierarchy and responsive portfolio design.
- Multiple exploratory visualizations plus actual-vs-predicted analysis.
- Direct comparison of feature groups and two model families.
- Chronological holdout design to reflect unseen future seasons.
- Error analysis and discussion of omitted baseball factors.
