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
aim up --host=0.0.0.0
```

Note down the port (e.g. `PORT=43800` for the address `http://0.0.0.0:43800`) and then open up a web browser on the local machine and navigate to `http://USERNAME@REMOTE_SERVER:PORT`. The Aim UI should now be accessible.
