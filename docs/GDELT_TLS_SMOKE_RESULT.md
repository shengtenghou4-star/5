# GDELT TLS Smoke Result

Source commit: `86647d6d3fb7e71111439d2fec0e81706c3d6c9d`

- tests: success
- real download: success

- requested: 3
- successful: 3
- restricted TLS fallbacks: 3
- country slice rows: [3, 3, 3]

| Requested | Observed | Status | Transport | Bytes | SHA-256 |
|---|---|---|---|---:|---|
| 2020-01-05T12:00:00+00:00 | 2020-01-05T12:00:00+00:00 | ok | official_tls_fallback | 67474 | `9630d8425ef85dad65aa669851064582f4da7e077eb0a6a2791756dc6a9ed308` |
| 2020-01-12T12:00:00+00:00 | 2020-01-12T12:00:00+00:00 | ok | official_tls_fallback | 76923 | `f1b401042a3863dc44fee0625b68b628164bc295b7f6430f011af852c1185012` |
| 2020-01-19T12:00:00+00:00 | 2020-01-19T12:00:00+00:00 | ok | official_tls_fallback | 46629 | `1d6048d89ddda9cb436b40a11c297a525cb2849dccb11a1bfd63fcd35889efbf` |

The fallback is restricted to certificate-verification failures on the exact official `data.gdeltproject.org` host. Cross-host redirects are rejected and every payload must parse as a GDELT export ZIP before caching.
