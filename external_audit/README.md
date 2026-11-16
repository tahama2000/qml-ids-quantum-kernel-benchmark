# External audits (case studies B and C)

Audits of two published QML-IDS benchmarks against the CRISP-QCML-IDS 14-item checklist and 13 review gates, with re-run scripts and per-seed results.

- `kerkatou2026_audit_v1/`: Kerkatou et al., IEEE Access 2026, doi:10.1109/ACCESS.2026.3693627 (code: github.com/raidanis/QML-IDS-NSLKDD @ 2ebad1c). Data: NSL-KDD KDDTrain+_20Percent, from that repository.
- `badami2026_audit_v1/`: Badami, IEEE Access 2026, doi:10.1109/ACCESS.2026.3679234 (code: github.com/Ali-Badami/Quantum-IDS @ 54dbe25). Data: HAI 22.04 (github.com/icsdataset/hai). Place test1–4.csv in a folder, and provide train_minmax.csv (per-feature min/max over train1–6) alongside them.

How each gate is applied and judged (Pass / Flag / Fail): see [`../GATES.md`](../GATES.md).

Each folder has `AUDIT_REPORT.md`, `scripts/` and `results/`.
