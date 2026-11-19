# Edge analysis for corrected-000323

## Scope and sources

This report traces the five images in conflict group `corrected-000323`
using:

- `artifacts/duplicate_aware_splits/split_manifest.csv`
- `artifacts/duplicate_aware_splits/duplicate_verification.json`
- `artifacts/duplicate_aware_splits/near_duplicate_candidates.csv`
- `artifacts/duplicate_aware_splits/near_duplicate_review/manual_review_decisions.csv`
- `artifacts/duplicate_aware_splits_corrected/conflict_review/corrected-000323_visual_inspection.md`

No duplicate records, labels, splits, source images, or checkpoints were
modified.

## Exact duplicate memberships

| Original class | Filename | Exact duplicate-group ID | Existing partition |
|---|---|---|---|
| Crack | `20240301_073625 - Copy.jpg` | `exact-000640` | train |
| Crack | `20240301_073625.jpg` | `exact-000640` | train |
| Pothole | `13 (2).JPG` | `exact-000991` | validation |
| Pothole | `13 (3).JPG` | `exact-000080` | train |
| Pothole | `13.jpg` | `exact-000991` | validation |

## Edge summary

### Group-forming edges

| Source filename | Destination filename | Edge type | Similarity | Manual review status | Source/destination labels | Result |
|---|---|---|---:|---|---|---|
| `20240301_073625 - Copy.jpg` | `20240301_073625.jpg` | Exact duplicate | Exact match | Exact metadata, no manual edge required | Crack/Crack | Valid exact relationship |
| `13 (2).JPG` | `13.jpg` | Exact duplicate | Exact match | Exact metadata, no manual edge required | Pothole/Pothole | Valid exact relationship |
| `13 (2).JPG` | `13 (3).JPG` | Perceptual near duplicate | Hamming distance 0 | `candidate-271`, `duplicate` | Pothole/Pothole | Visually supported |
| `13 (3).JPG` | `13.jpg` | Perceptual near duplicate | Hamming distance 0 | `candidate-272`, `duplicate` | Pothole/Pothole | Visually supported |
| `20240301_073625.jpg` | `13.jpg` | Perceptual near duplicate | Hamming distance 8 | `candidate-156`, `duplicate` | Crack/Pothole | **False positive; inconsistent with the visual inspection** |

The exact duplicate edge `exact-000640` has no hamming-distance score because
it represents byte-level content equivalence in the exact metadata.
The exact duplicate edge `exact-000991` likewise has no hamming-distance
score.

## Complete transitive path

The component `corrected-000323` is created by the following path:

```text
Crack: 20240301_073625 - Copy.jpg
  |
  | exact duplicate edge: exact-000640
  |
Crack: 20240301_073625.jpg
  |
  | candidate-156
  | manual_decision = duplicate
  | edge type = perceptual near duplicate
  | hamming_distance = 8
  |
Pothole: 13.jpg
  |
  | exact duplicate edge: exact-000991
  |
Pothole: 13 (2).JPG
  |
  | candidate-271
  | manual_decision = duplicate
  | edge type = perceptual near duplicate
  | hamming_distance = 0
  |
Pothole: 13 (3).JPG
```

An alternate last leg is present directly in the Pothole triplet:

```text
13 (3).JPG
  |
  | candidate-272
  | manual_decision = duplicate
  | hamming_distance = 0
  |
13.jpg
```

Together, the exact and confirmed near-duplicate edges connect both exact
Pothole members and the singleton Pothole image to the Crack pair.

## All relevant reviewed cross-class edges

These edges have manual review decisions but do not form a duplicate
relationship. They therefore did not contribute to `corrected-000323`.

| Source filename | Destination filename | Candidate | Hamming | Review status |
|---|---|---|---:|---|
| `20240301_073625 - Copy.jpg` | `13 (2).JPG` | `candidate-150` | 8 | `not_duplicate` |
| `20240301_073625 - Copy.jpg` | `13.jpg` | `candidate-151` | 8 | `not_duplicate` |
| `20240301_073625.jpg` | `13 (2).JPG` | `candidate-155` | 8 | `not_duplicate` |
| `20240301_073625.jpg` | `13.jpg` | `candidate-156` | 8 | **`duplicate`** |

There is no manual reviewed cross-class edge between either Crack image and
`13 (3).JPG`. The near-duplicate candidate file does contain two unreviewed
same-partition candidate links:

- `20240301_073625 - Copy.jpg` to `13 (3).JPG`, hamming distance 8
- `20240301_073625.jpg` to `13 (3).JPG`, hamming distance 8

These were not manually reviewed and did not contribute to the transitive
component.

## Inconsistent connection

### False-positive duplicate edge

**Candidate: `candidate-156`**

- Source: `20240301_073625.jpg`
- Destination: `13.jpg`
- Edge type: Perceptual near duplicate
- Hamming distance: 8
- Manual status: `duplicate`
- Source original label: Crack
- Destination original label: Pothole

The visual inspection found that the source and destination show different
road scenes:

- The source shows a smooth gray asphalt surface with a distributed hairline
  crack network.
- The destination shows a rough aggregate surface with a localized narrow
  damaged area.
- The two images do not appear to represent the same physical damage location,
  angle, or crop.

Therefore, `candidate-156` is the link inconsistent with the visual inspection
and is identified as the false-positive connection between the Crack pair and
the Pothole triplet.

## Preserved valid relationships

The following relationships remain valid and should not be removed from the
duplicate metadata:

1. Exact duplicate: `20240301_073625 - Copy.jpg` and
   `20240301_073625.jpg`, in `exact-000640`.
2. Exact duplicate: `13 (2).JPG` and `13.jpg`, in `exact-000991`.
3. Confirmed near duplicate: `13 (2).JPG` and `13 (3).JPG`,
   `candidate-271`, hamming distance 0.
4. Confirmed near duplicate: `13 (3).JPG` and `13.jpg`,
   `candidate-272`, hamming distance 0.

These four relationships keep the two visual scenes internally coherent:
Crack pair and Pothole triplet.

## Conclusion

The false-positive edge is `candidate-156`:

```text
20240301_073625.jpg -- hamming 8, duplicate --> 13.jpg
```

Removing this cross-class grouping edge would separate the current
transitive component into the two visually distinct scenes. No duplicate
records or labels were changed while preparing this report.
