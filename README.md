# MoE Competition Docs

Read the Docs source for the MoE load-balancing competition background material.

The documentation covers:

- DeepSeek model-family overview and architecture
- Mixture-of-Experts concepts
- DeepSeekMoE routing and load-balancing mechanisms
- How this background relates to the competition simulator

## Local build

```bash
python -m pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```
