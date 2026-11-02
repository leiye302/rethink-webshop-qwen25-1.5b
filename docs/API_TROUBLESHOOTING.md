# API connection checks and failure handling

This is transport engineering, not an algorithm or prompt change. Both backbone
sizes use identical code. The model remains `glm-5.3-flash`, Actor marker IDs alone
select local-hint targets, hints go only to the current-Actor Teacher, and SDAR's
losses, gate beta5, training settings and fixed140 evaluation remain unchanged.

## Check before a GPU job

Use the same `HINT_API_BASE`, `HINT_MODEL` and `HINT_API_KEY` environment settings
as training. Do not put a key in a command, report, screenshot or Git commit.
If `RETHINK_HINT_AUTH` is already set, the owner-only private file takes precedence.

```bash
export HINT_API_BASE="https://opencode.ai/inference/go/openai/v1"
export HINT_MODEL="glm-5.3-flash"
read -rsp 'Hint API key: ' HINT_API_KEY; echo
export HINT_API_KEY

# Local configuration and actual cache write check; no HTTP request, CUDA or weights.
PYTHONPATH="$PWD/src" python -m rethink_webshop.hint_api --check

# Opt in to one fresh synthetic request using the real training body/transport/parser.
PYTHONPATH="$PWD/src" python -m rethink_webshop.hint_api --check --allow-api-cost
```

These checks need only the repository and Python3.10+, not SDAR, Java, Torch or
GPU installation. The paid check uses one small synthetic shopping step and no
request retries; every invocation uses a fresh probe directory, so a previous
cached success cannot hide a new outage. It checks transport/protocol, not the
semantic quality of hints or benchmark performance. Probe responses remain in
ignored `runs/hint_api_check`, not in the source distribution.

For a compatible non-Go provider, use its HTTPS OpenAI-compatible base and the
same fixed model and request parameters. **Base ends before `/chat/completions`**;
the code appends that suffix. A full completions URL, plaintext HTTP, credentials
in the URL, malformed port or key-bearing query is rejected before a request.
No endpoint/model substitution or automatic paid-provider fallback is added.

## Proxies and TLS

Default behavior is unchanged: urllib inherits the current environment/system
proxy settings. An old proxy can break a new server's requests; another server
may require its proxy. Never assume one global answer.

```bash
# Optional, deliberate direct connection. Applies to the check and Ray training driver.
export RETHINK_HINT_TRUST_ENV=false
```

Omit that variable or use `true` to preserve the default proxy behavior. Only
`true`, `false`, `1`, `0` are accepted. The launcher forwards the resolved boolean
to Ray; no proxy address or credential is printed or added to Hydra. There is no
automatic retry through a different network route. TLS verification stays on and
redirects stay refused, so a retry cannot forward the API key to another host.

## Understand the error

Logs contain only reason codes, request SHA, attempt count, elapsed time, stage,
numeric status/errno and a fixed error category. They never stringify exceptions
or print Authorization, response error bodies, private keys or proxy URLs.

| Signal | Meaning / operator action |
|---|---|
| `stage=open`, `error_kind=dns_error` | DNS resolution failed; check the server resolver/network |
| `stage=open`, `timeout` or `remote_disconnected` | Failure before response headers; connection, proxy, TLS or server wait may be involved |
| `stage=read`, `timeout` / `incomplete_response` | Failure while reading the response body |
| `stage=decode`, `api_invalid_json` | HTTP response was not valid JSON; not a cache or routing error |
| `http_401` / `http_403` | Check credentials, provider access/credit policy; permanent HTTP failure stops |
| `http_400` / `http_404` | Check base path/model/parameter compatibility; do not blindly retry |
| `http_429` / selected `5xx` | Transient quota/service issue; retained-batch retry |
| `api_tls_certificate_error` | Fix certificate/CA/network interception; do not disable TLS verification |
| `api_cache_*` / `api_status_write_error` | Fix local disk space, cache permissions or corruption; not an API connection failure |

`open` includes DNS/connection/TLS and waiting for response headers. These logs
do not claim to measure each of those phases separately. The same user-visible
timeout from a different machine is not proof of an identical root cause.

## Retry and no-silent-fallback contract

The fixed timeout remains120 seconds per blocking socket operation, retries5
(6 attempts), with backoff2/4/8/16/32 seconds. This is **not** a strict request or
batch-wide wall-clock deadline:

\[
T_{\mathrm{nominal}}=6\times120+(2+4+8+16+32)=782\;\mathrm{s}.
\]

Transient failures still retain exactly the same rollout/pre-update Actor and
wait60 seconds before another batch attempt. They never silently disable hints
or update the Actor with a different intervention. Successful episodes remain
in memory and on disk when a different episode retries.

Permanent HTTP/configuration/certificate and cache I/O failures stop with a
clear error instead of an infinite five-minute retry loop. A local cache-write
failure after a successful HTTP response does not reissue the paid request.
An unwritable status/cache directory is checked before the batch starts HTTP.

The concurrent collector consumes completed tasks without submission-order
blocking, immediately reports failures, cancels queued requests and joins
already-running requests. There is never an orphan retry pool running beside
a new one. In-flight urllib calls cannot be forcibly interrupted; waiting is
visible through approximately30-second batch heartbeats and
`runs/hint_cache/api_wait_status.json`. A stopped process cannot durably retain
an in-memory rollout; continuation remains subject to the existing complete
native checkpoint/resume contract.

Malformed/illegal *completed hint content* still uses the ordinary static-skill
Teacher with recorded reason, exactly as before. Neither that parser nor the
policy of rejecting a truncated Teacher hint is changed.

## Apply updates safely

Do not update files under a live trainer or delete/modify its launch receipt.
The launch identity binds hint source hashes, so an old job is not silently
declared to be a same-code resume after this update. Fresh jobs use a new run
directory. Migrating an existing job requires an explicit reviewed migration;
do not bypass the identity or discard optimizer state merely to make it start.

CPU tests cover simulated timeouts, HTTP errors, disk failures, proxy modes,
cache reuse, retained-batch retry and diagnostic privacy. Passing those tests
and a real API probe does not establish GPU/Ray training or long-run service
availability; use your actual server's check before an unattended job.
