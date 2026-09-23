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
- **Training:** 2000–2022
- **Testing:** 2023–2025

A chronological split was selected so the final seasons are genuinely unseen during model training and to reduce temporal leakage.

KNN uses standardization because it is distance-based. The project establishes a mean-wins baseline and compares models using MAE, RMSE, and R².

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
The results suggest that both run creation and run prevention provide useful predictive information about team wins. The project does **not** establish that pitching or hitting causes wins; it evaluates predictive performance using the selected statistics and seasons.

## Limitations
The model does not include every factor that can affect wins, including defense, baserunning, injuries, roster construction, payroll, strength of schedule, park effects, and bullpen usage. The test period contains only three seasons, so performance may differ on another time period.

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
