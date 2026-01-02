# Troubleshooting

## Common Issues

### MLflow Connection Issues

Ensure MLflow server is running:

```bash
make mlflow-ui
```

### AWS Credentials

Verify AWS credentials are configured:

```bash
aws sts get-caller-identity
```

### Dependency Issues

Reinstall dependencies:

```bash
make setup
```

### Test Failures

Run tests with verbose output:

```bash
uv run pytest tests/ -v
```
