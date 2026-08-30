# Hermes Agent Approval Bridge

An Open WebUI Pipe Function that connects to [Hermes Agent][hermes]'s structured
runs API and turns `approval.request` events into attended **Allow once / Deny**
dialogs. It complements Open WebUI's [standard Hermes connection guide][guide]
when an attended approval surface is required.

The bridge is intentionally fail-closed. A missing browser connection, a
timeout, a malformed event, or a failed approval response never becomes
consent. The bridge denies the pending operation when possible and stops a run
whose state cannot be reconciled.

## Requirements

- Open WebUI 0.11.1 or newer
- Hermes Agent with these `/v1/capabilities` features:
  - `run_submission`
  - `run_events_sse`
  - `run_approval_response`
  - `approval_events`
- A Hermes `API_SERVER_KEY`
- A live browser tab for attended approvals

## Install

1. In Open WebUI, open **Admin Panel → Functions → Create**.
2. Paste the contents of `main.py`, save it, and enable the Function.
3. Configure its valves:
   - `HERMES_BASE_URL`: for the shared Docker network, typically
     `http://hermes:8642/v1`
   - `HERMES_API_KEY`: the value Hermes uses as `API_SERVER_KEY`
   - `MODEL_ID`: normally `hermes-agent`
4. Select **Hermes Agent (Approvals)** in the model picker.

Do not place provider credentials such as `OPENROUTER_API_KEY` or
`OPENAI_API_KEY` in `HERMES_API_KEY`. This valve is only the inbound bearer
between Open WebUI and Hermes.

## Approval behavior

- **Allow once** posts `{"choice": "once"}` to the run-scoped Hermes approval
  endpoint.
- **Deny**, dialog failure, browser disconnect, or timeout posts
  `{"choice": "deny"}`.
- JSON approval payloads are indented in a fenced preview, while live tool
  status reduces structured payloads to a short key/value summary.
- Session and permanent approval choices are deliberately not offered in the
  initial version.
- The command preview is the redacted value emitted by Hermes. The bridge does
  not request or display raw secrets.

Open WebUI's confirmation events require a live WebSocket/browser connection.
Set `APPROVAL_TIMEOUT_SECONDS` to a finite attended-operation window. API,
automation, and background use should have a separate non-interactive policy;
silence is never treated as approval.

## Development

From the repository root:

```sh
python -m unittest discover \
  -s functions/pipes/hermes_agent_approvals/tests \
  -p 'test_*.py'
```

The runtime dependency is `httpx`; Open WebUI supplies `pydantic`.

## Upstream shape

The implementation is a provider Pipe rather than an Open WebUI core patch.
That keeps the standard OpenAI connection untouched and uses Open WebUI's
documented plugin event seam for Hermes' structured runs and approval protocol.

[hermes]: https://github.com/NousResearch/hermes-agent
[guide]: https://docs.openwebui.com/getting-started/quick-start/connect-an-agent/hermes-agent/
