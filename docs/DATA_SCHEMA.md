# Data Schema

All processed examples use JSON Lines with the following required fields:

| Field | Description |
| --- | --- |
| `text_t` | Natural-language current-state description |
| `action_text` | Natural-language action or event |
| `text_t1` | Natural-language next-state description |

Optional fields include `id`, `split`, `dataset`, `transition_type`,
`attribute`, `entity`, `episode`, or other source-specific metadata.

The official retrieval protocol is row-level. Exact duplicate triples
`(text_t, action_text, text_t1)` are removed during preprocessing, but examples
that share the same next-state text while differing in current state, action,
entity, or trajectory context are retained.
