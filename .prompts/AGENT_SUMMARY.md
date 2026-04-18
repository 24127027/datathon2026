# Datathon 2026 Agent Handoff

## Purpose
This document gives an implementation-ready summary of the current workspace so another agent can continue work quickly.

Constraint used for this summary:
- Raw dataset content was not inspected.
- Summary is based only on project structure and existing source code.

## Core Requirement
Build a reproducible, code-first baseline pipeline for this Datathon workspace without inspecting raw dataset values unless explicitly requested.

The pipeline must:
- Load data through reusable Python functions.
- Perform deterministic preprocessing and table integration.
- Generate a prediction-ready artifact aligned to competition submission format.
- Be runnable from a script entrypoint, not only notebooks.

## Workspace Snapshot
- Root notebooks/scripts:
  - `base.ipynb`
  - `main.py` (currently empty)
- Project config:
  - `pyproject.toml`
  - `README.md` (currently empty)
- Source package:
  - `src/data_utils.py`
- Dataset root:
  - `data/datathon-2026-round-1/`

## Data Layout (File-Level)
- `analytical/`
  - `sales.csv`
- `master/`
  - `customers.csv`
  - `geography.csv`
  - `products.csv`
  - `promotions.csv`
- `operational/`
  - `inventory.csv`
  - `web_traffic.csv`
- `transaction/`
  - `orders.csv`
  - `order_items.csv`
  - `payments.csv`
  - `returns.csv`
  - `reviews.csv`
  - `shipments.csv`
- Other:
  - `baseline.ipynb`
  - `sample_submission.csv`

## Existing Code Behavior
The file `src/data_utils.py` provides dataset loader helpers with light joins:

1. `load_customers`
- Loads `master/customers.csv`.

2. `load_products`
- Loads `master/products.csv`.

3. `load_promotions`
- Loads `master/promotions.csv`.

4. `load_orders`
- Loads `transaction/orders.csv`, `payments.csv`, `shipments.csv`.
- Drops `payment_method` from payments if present.
- Left-joins payments to orders on `order_id`.
- Filters shipments to orders whose `order_status` is in: `shipped`, `delivered`, `returned`.
- Left-joins filtered shipments to the merged orders+payments table.

5. `load_order_items`
- Loads `transaction/order_items.csv`, `returns.csv`, `reviews.csv`.
- Drops `customer_id` from reviews if present.
- Left-joins returns on `order_id, product_id`.
- Left-joins reviews on `order_id, product_id`.

6. `load_inventory`
- Loads `operational/inventory.csv`.

7. `load_web_traffic`
- Loads `operational/web_traffic.csv`.

8. `load_sales`
- Loads `analytical/sales.csv`.

## Requirements For Next Agent

### Must-Have Requirements
1. Objective clarification
- Determine target and prediction grain from challenge materials and existing project assets.
- Do not infer target from random raw-value exploration.

2. Data access layer
- Keep all table access centralized in `src/data_utils.py`.
- Add missing loader for `master/geography.csv`.
- Maintain `data_root` configurability for every loader.

3. Reproducible pipeline
- Implement `main.py` as executable pipeline entrypoint.
- Ensure the run is deterministic (fixed seeds if randomness is used).
- Produce outputs to a clear folder (for example `outputs/`).

4. Data quality validation
- Add checks for null join keys.
- Add checks for unexpected duplicate keys where keys should be unique.
- Add row-count sanity checks around critical joins.

5. Submission compatibility
- Ensure output schema matches the competition submission structure.
- Save final submission file in CSV format.

6. Documentation and usability
- Update `README.md` with setup, run command(s), and assumptions.
- Document what each major script/module is responsible for.

### Should-Have Requirements
- Add `load_all_tables()` helper returning a dictionary of DataFrames.
- Add a lightweight schema/contract module (for example `src/schema.py`) for expected keys and join logic.
- Add smoke tests for loader availability and essential columns.

### Out Of Scope (Unless Explicitly Requested)
- Deep model optimization or hyperparameter search.
- Heavy feature engineering experiments.
- Inspecting or publishing raw sensitive values.

## Environment
From `pyproject.toml`:
- Python: `>=3.13`
- Dependencies:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `sqlalchemy`


## Guardrails For Next Agent
- Do not inspect raw dataset values unless explicitly requested.
- Keep joins explicit and reversible.
- Prefer small, testable functions over notebook-only logic.
- Keep all paths configurable via `data_root`.
