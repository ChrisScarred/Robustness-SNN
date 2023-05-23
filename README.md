# Robustness SNN
Testing the robustness of SNN-based spoken digits classification.

## Instructions

1. Copy `config/example.yaml` into a different .yaml file and fill in the required configurations.
2. Copy `example.env` to `.env` and fill in the location to your config file.
3. Install the dependencies through `pip install -r requirements.txt` (preferably in a virtual environment).
4. *(If necessary, install Redis)* Start the Redis server.
5. Run via `python -m src.main`.

## Notes

- Redis will store the cached data for 3 days on default.
