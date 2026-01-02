# Architecture Design

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      SageMaker                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  Preprocess │──│    Train    │──│  Evaluate   │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│         │                │                │                  │
│         └────────────────┼────────────────┘                  │
│                          │                                   │
│                    ┌─────▼─────┐                             │
│                    │   MLflow  │                             │
│                    │  Tracking │                             │
│                    └───────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Steps

Modular pipeline steps for each stage of the ML workflow.

### Pipelines

Orchestration of steps into complete workflows.

### Utilities

Shared helper functions and configurations.
