# CompressAI Trainer

An easy way to train and evaluate CompressAI-based models.

## Installation

```bash
pip install poetry
poetry install
poetry run pip install -e /path/to/compressai
```

## Running

```bash
poetry run python train.py --config-path="conf" --config-name="example"
```

## Log viewers

### Aim

Aim logs all experiments to a single directory containing an `.aim` repository.

#### Local-only

If the logs are available in a locally accessible directory, simply navigate to the directory containing the `.aim` repository and run:
```bash
aim up
```

#### Remote host

If the logs are on a remote host, then on the remote host, navigate to the directory containing the `.aim` repository and run:
```bash
aim up
```

Note down the address (e.g. `http://127.0.0.1:43800`) and then on the local machine, run:
```bash
ssh -L 8080:127.0.0.1:43800 USERNAME@REMOTE_SERVER
```

Now open up a web browser on the local machine and navigate to `http://localhost:8080`. The Aim UI should now be accessible.

