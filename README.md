Documentation TBD!!

## CI

This repository uses Woodpecker CI on Codeberg via [`.woodpecker.yml`](.woodpecker.yml).

- The `test` step runs on `push`, `pull_request`, and manual pipeline runs.
- The `publish` step runs when a tag matching `v*` is pushed.

### Codeberg setup

1. Request and enable CI access for the repository in `https://ci.codeberg.org/repos/add`.
2. Add a repository secret named `pypi_token`.
3. Use a PyPI API token for that secret value.

GitHub's previous PyPI trusted publishing flow does not carry over to Woodpecker, so publishing now uses the `pypi_token` secret with `uv publish`.
