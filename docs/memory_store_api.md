# Memory Store API — contract

Server side of `DynamiqMemoryStore` (`dynamiq/storages/memory/dynamiq.py`), which gives an agent
notes it keeps across conversations. Short text memories addressed by path, scoped to a store and a
user. No search, no versioning, no binary content.

A conforming reference implementation lives in
`tests/integration_with_creds/agents/test_agent_memory_store.py` as `MemoryStoreAPISimulator`, with
`test_store_crud_against_the_api` and `test_store_error_branches` as the conformance suite — run
those two first against a real server, since they exercise this contract with no model involved.

## Conventions

| | |
|---|---|
| **Base** | `{DYNAMIQ_URL}/v1/memory-stores/{store_id}/files` |
| **Auth** | `Authorization: Bearer <api_key>` (existing Dynamiq API key) |
| **Sent** | `Content-Type: application/json` on every request |
| **Envelope** | every success response is `{"data": ...}` |
| **Content** | a plain JSON string — memories are text, so there is no base64 |

`user_id` is **required on every call** — query string for GET and DELETE, body for PUT. It
identifies the end user *within* a store. The agent never supplies it and cannot override it, so
enforce isolation from it together with the credentials.

`path` is an opaque key. The client sends a normalized relative path (`preferences.md`,
`team/naming.md` — never leading-slashed) and expects the identical string back. Do not normalize or
re-root it.

On **list**, `path` is a **key prefix**, not a directory: match it with `LIKE 'team/%'` rather than
walking a tree. There is no `recursive` flag because a prefix match is already transitive, and no
glob `pattern` — memories are notes with descriptive paths, so filtering them server-side buys
nothing. `path=""` returns every memory for that store and user.

## Where each parameter goes

GET and DELETE carry no body, so their parameters ride in the query string. PUT has a body, and a
memory can be long.

| endpoint | path | query | body |
|---|---|---|---|
| `GET /files` | `store_id` | `path`, `user_id` | — |
| `GET /files/content` | `store_id` | `path`, `user_id` | — |
| `PUT /files` | `store_id` | — | `path`, `content`, `user_id` |
| `DELETE /files` | `store_id` | `path`, `user_id` | — |

---

## 1. List — `GET /files`

**Query:** `path` (a key prefix; empty lists everything) · `user_id`

Returns metadata only, **never `content`**:

```json
{ "data": [
  { "path": "preferences.md", "size": 24, "updated_at": "2026-01-15T10:30:00Z" }
] }
```

`size` and `updated_at` are optional to the client — a missing `size` falls back to the content
length and an unparseable `updated_at` becomes null. An empty store is `{"data": []}`, not a `404`.

## 2. Read — `GET /files/content`

**Query:** `path` · `user_id`

```json
{ "data": { "path": "preferences.md", "content": "Prefers British English.",
            "size": 24, "updated_at": "2026-01-15T10:30:00Z" } }
```

An empty or absent `data` is read as "not found" and raises, so return `404` for a missing path
rather than `200` with a null body. This is the hot path — an agent lists and reads at the start of
most conversations.

## 3. Write — `PUT /files`

**Body:**

```json
{ "path": "preferences.md", "content": "Prefers British English.", "user_id": "u-42" }
```

Returns the stored entry, same shape as a list row. **Write is an upsert** — it always replaces, so
there is no `overwrite` flag and no `409`. Editing part of a memory is read-modify-write on the
client: it reads, replaces the text, and writes the whole memory back.

## 4. Delete — `DELETE /files`

**Query:** `path` · `user_id`

```json
{ "data": { "deleted": true } }
```

`204` with an empty body is accepted. `404` is read as "was not there" and returns false rather than
raising.

---

## Status codes

| code | meaning | what the client does |
|---|---|---|
| `2xx` | ok | parses `data` |
| `404` | no such path | read raises `MemoryNotFoundError`; delete returns `false` |
| `403` | not permitted | raises `MemoryPermissionError` |
| any other `≥400` | including `413` for oversize | raises `MemoryStoreError` with the status and body |
| non-JSON body | — | raises `MemoryStoreError` |

Error bodies are logged verbatim, so a readable `message` helps — no error shape is required.

## Server requirements

- **Reject path escapes server-side** — `..`, absolute paths, URL-encoded traversal (`%2e%2e%2f`).
  Do not rely on client validation.
- **Cap memory size and per-store count**; return `413`.
- `updated_at` is ISO-8601 (a trailing `Z` is handled).
- Reads are frequent, writes are rare.

## Not in scope

Semantic search, versioning or history, rename, partial writes, and binary content. Rename is
delete + write.
