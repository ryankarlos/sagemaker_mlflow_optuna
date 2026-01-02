# Running Pipelines

## FM Optuna Pipeline

Run hyperparameter optimization for Factorization Machines:

```bash
# Quick local test
make pipeline-fm

# Full training
make pipeline-fm-full

# Or directly with options
python -m pipelines.fm_optuna_train \
    --local \
    --n_users 5000 \
    --n_games 100 \
    --max_trials 30 \
    --experiment_name my-experiment
```

### Options

- `--n_users` - Number of users to simulate (default: 5000)
- `--n_games` - Number of games to simulate (default: 100)
- `--n_days` - Days of data to simulate (default: 180)
- `--train_days` - Days for training split (default: 150)
- `--max_trials` - Max Optuna trials (default: 20)
- `--early_stopping` - Early stopping rounds (default: 5)
- `--experiment_name` - MLflow experiment name
- `--local` - Use local FM simulator (no SageMaker)
- `--seed` - Random seed (default: 42)

## Viewing Results

Start MLflow UI to view experiment results:

```bash
make mlflow-ui
```

Then open http://localhost:5000
