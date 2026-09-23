# MLB Hitting vs Pitching: Predicting Team Wins

## Research Question
How well can MLB team wins be predicted using hitting statistics compared with pitching/run-prevention statistics?

## Project Overview
This machine-learning project uses MLB team-season data from 2000–2025 to predict team wins. It compares models built from hitting features, pitching features, and a combined feature set.

### Target
- **Wins (W)** — regression target

### Hitting features
- Runs per game
- On-base percentage (OBP)
- Slugging percentage (SLG)
- Home runs per game

### Pitching features
- Runs allowed per game
- ERA
- Strikeouts per 9 innings

### Validation strategy
Training data: 2000–2022
Testing data: 2023–2025

A chronological split is used so that later seasons are held out as genuinely unseen data.

## Preliminary Results
The preliminary analysis found that pitching-only models predicted wins somewhat better than hitting-only models on the held-out 2023–2025 seasons. The combined models performed best, indicating that both offensive and run-prevention information contributes to predicting wins.

## Repository Files
- `project2_mlb_hitting_vs_pitching.py` — starter analysis code

## Data Source
Lahman Baseball Database / Teams dataset: https://github.com/corbtastik/lahman-baseball-db

## Transparency
Generative AI was used to assist with project planning, code structure, data analysis, and written explanations. All analysis and final interpretations should be reviewed and understood by the student before submission.
