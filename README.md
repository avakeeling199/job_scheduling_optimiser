# 🧬 Multi-Objective Genetic Algorithm for Workforce Scheduling

This project implements a multi-objective genetic algorithm (NSGA-II) to optimise job assignments in a workforce scheduling problem. The two objectives are:

- 🧑‍💼 **Minimise** the number of distinct workers required  
- ⚖️ **Maximise** fairness, measured as the negative variance in workload distribution

The problem is inspired by real-world scenarios where workers have limited availability and qualifications, and tasks are time-bound.

---

## 📄 Report

📘 **COMP5012 Coursework Report**  
Submitted as part of the MSc in Health Data Science — University of Plymouth (2025)

See `report.pdf` for full methodology, parameter tuning process, results, Pareto front analysis, and validity checks.

---

## 🔧 How to Run the Algorithm

### ✅ Final run (using selected parameters)

```bash
python algorithm_main.py
```

Make sure `final_run_only = True` in `algorithm_main.py` to run only the best configuration:
- Population size: 100
- Mutation probability: 0.2
- Generations: 100
- Penalty weight (λ): 10

This will:
- Run NSGA-II with the selected parameters
- Save convergence plots and Pareto front visuals
- Output a validity-checked summary of the final archive

---

### 🔄 Full parameter tuning (optional)

To reproduce the parameter grid search:

```python
final_run_only = False
```

This will iterate over combinations of population, mutation rate, λ, and generations, and save: 
- Plots for each configuration
- A `summary_results.csv` file for comparison

### 📁 File Overview

### 📁 File Overview

| File                    | Description                                         |
|-------------------------|-----------------------------------------------------|
| `algorithm_main.py`     | Main optimisation script with NSGA-II and plotting |
| `data_parser.py`        | Loads and structures the input `.dat` file         |
| `summary_results.csv`   | CSV of all tuning runs for comparison              |
| `final_run_results.csv` | Validity-checked results from the final run        |
| `report.pdf`            | Final submitted report (if included)               |

---

### 🧹 Repo Notes

-`.DS_Store` and plot images are excluded from version control
- Requires Python 3.11+ and the following libraries: 
    - `deap`
    - `numpy`
    - `matplotlib`
    - `pandas`

### 📬 Contact

**Author**: Ava Keeling

**Email**: ava.keeling@students.plymouth.ac.uk

**Course**: MSc Health Data Science

**Module**: COMP5012 - Computational Intelligence
