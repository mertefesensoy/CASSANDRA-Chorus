# CASSANDRA Chorus

CASSANDRA Chorus is the next phase of the [CASSANDRA](https://github.com/mertefesensoy/CASSANDRA) low-hardware language-model research program. The long-term goal is a language model trained by a pool of volunteer machines, where no machine holds the whole model during training, and where the full pipeline, logs and weights are open. No token or cryptocurrency is involved; contribution is tracked in a plain credit ledger.

The model is a mixture-of-experts transformer. Each worker trains the shared part of the model plus a subset of the experts locally, and a coordinator merges the results ("sliced local training"). Whether this works for a language model is the open question that Stage 0 tests, on a single machine, before any pool infrastructure is built.

## Status

Planning. Nothing is implemented yet, and no results exist.

## Documents

- [`docs/SRS.md`](docs/SRS.md): software requirements specification
- [`docs/PLAN.md`](docs/PLAN.md): plan memo, stages and Stage 0 work breakdown

## Licence

Apache License 2.0. See [`LICENSE`](LICENSE).
