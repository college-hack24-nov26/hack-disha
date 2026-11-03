# Personalization data root

Runtime doctor profiles, verified samples, adapters, and audits are stored under
`doctors/<doctor_id>/`.

This tree is **gitignored** for privacy. See `docs/DOCTOR_PERSONALIZATION.md`.

Create a profile:

```text
.\.venv\Scripts\python.exe scripts\doctor_personalization_cli.py create-profile --doctor-id doc_demo
```
