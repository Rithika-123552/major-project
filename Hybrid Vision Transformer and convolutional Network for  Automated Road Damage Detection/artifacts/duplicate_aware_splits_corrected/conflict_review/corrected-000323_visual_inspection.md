# Visual inspection report: corrected-000323

## Scope

The following analysis is based on the contact sheet
`corrected-000323_contact_sheet.jpg`. It is not based on any automatic label
resolution or modification of the source images.

## Summary

The five images appear to contain two different physical road-damage scenes:

- `20240301_073625 - Copy.jpg` and `20240301_073625.jpg` show the same
  smooth, weathered asphalt surface and the same visible crack pattern.
- `13 (2).JPG`, `13 (3).JPG`, and `13.jpg` show the same rough road surface
  and the same localized cracked or pitted area.

The top pair should not be treated as the bottom triplet or vice versa. The
top and bottom scenes do not appear to depict the same physical location.

## Individual image analysis

### 1. 20240301_073625 - Copy.jpg

- **Complete filename:** `20240301_073625 - Copy.jpg`
- **Original class label:** Crack
- **Original dimensions:** 4032 x 3024 pixels
- **Visible observations:**
  - The image shows a smooth gray asphalt surface.
  - Several irregular, dark, hairline cracks run across the surface.
  - No clearly defined hole or displaced road area is visible.
  - The visible pattern is dominated by thin linear fracture lines rather
    than a depression.
- **Visual classification:** Crack
- **Ambiguity:** Low ambiguity. The visible feature is visually consistent
  with cracking; no convincing pothole boundary is present.

### 2. 20240301_073625.jpg

- **Complete filename:** `20240301_073625.jpg`
- **Original class label:** Crack
- **Original dimensions:** 4032 x 3024 pixels
- **Visible observations:**
  - The road surface and crack arrangement are visually identical to
    `20240301_073625 - Copy.jpg`.
  - The same irregular network of thin dark cracks is visible.
  - There is no visible enclosed, depressed hole.
- **Visual classification:** Crack
- **Ambiguity:** Low ambiguity.

### 3. 13 (2).JPG

- **Complete filename:** `13 (2).JPG`
- **Original class label:** Pothole
- **Original dimensions:** 495 x 264 pixels
- **Visible observations:**
  - The image shows a rough, light-gray aggregate road surface.
  - A narrow dark linear feature runs through the center-right area.
  - At least one small irregular dark patch or depression is visible where
    the surface breaks.
  - The image is much more tightly cropped than `13 (3).JPG` and `13.jpg`.
- **Visual classification:** Uncertain
- **Ambiguity:** The feature appears both elongated and pitted. It could be a
  crack with localized surface loss, or a small pothole, so it is not possible
  to classify it confidently from this view alone.

### 4. 13 (3).JPG

- **Complete filename:** `13 (3).JPG`
- **Original class label:** Pothole
- **Original dimensions:** 512 x 512 pixels
- **Visible observations:**
  - The same rough aggregate surface is visible as in the other `13` images.
  - The view includes the entire localized dark feature and more surrounding
    road surface.
  - The feature is a narrow, irregular depression or break in the road,
    rather than a broad circular pothole.
  - The narrow shape suggests a crack or severely damaged joint, although
    some loss of surface material may be present.
- **Visual classification:** Uncertain
- **Ambiguity:** The location and extent of the feature support possible
  surface cracking, but the image does not establish a clear pothole boundary.

### 5. 13.jpg

- **Complete filename:** `13.jpg`
- **Original class label:** Pothole
- **Original dimensions:** 495 x 264 pixels
- **Visible observations:**
  - This is visually highly similar to `13 (2).JPG`.
  - It shows the same narrow dark linear feature and small irregular damaged
    patch on rough aggregate.
  - The crop is substantially narrower than `13 (3).JPG`, but the visible
    physical feature appears the same.
- **Visual classification:** Uncertain
- **Ambiguity:** The view may show the same crack or small depression as
  `13 (2).JPG` and `13 (3).JPG`, but a definitive pothole classification
  cannot be established.

## Pair and group comparisons

### Crack-scene pair

`20240301_073625 - Copy.jpg` and `20240301_073625.jpg` appear identical at the
contact-sheet level. They have the same dimensions and display the same crack
network. This is therefore consistent with the same physical damage captured
in effectively identical copies or views, not two distinct instances.

### 13-image triplet

`13 (2).JPG`, `13 (3).JPG`, and `13.jpg` appear to show the same rough road
surface and the same localized narrow damaged area. The differences visible in
the contact sheet are consistent with crops or different image presentation:

- `13 (2).JPG` and `13.jpg` have the same dimensions and appear highly similar,
  possibly identical crops.
- `13 (3).JPG` has a square crop and shows more of the surrounding road,
  consistent with the same area viewed at a different framing or crop.

This is consistent with the same physical damage, not three unrelated road
locations.

### Comparison between the two scenes

The top pair and bottom triplet differ in:

- Surface texture
- Viewing field and crop
- Crack/pit shape
- Overall road appearance

The top pair shows a smooth gray surface with a distributed hairline crack
network. The bottom triplet shows rough aggregate with one localized narrow
feature. These observations do not indicate the same physical location or
damage instance.

## Conclusions

1. The two `20240301_073625` images visually show the same crack scene and are
   likely the same physical damage or duplicate views.
2. The three `13` images visually show the same narrow damaged area and are
   likely crops or views of the same physical instance.
3. The top pair and bottom triplet appear to be different physical damage
   instances.
4. The top pair is visually more consistent with a crack.
5. The three bottom images are ambiguous and are therefore marked uncertain;
   the local feature may be a crack or a small pothole.
6. This visual inspection supports a transitive conflict in the grouping path,
   but it does not automatically reassign either original class label.
