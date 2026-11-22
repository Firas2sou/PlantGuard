# Dataset audit - PlantVillage

- Root: `C:\plant-doctor\data\raw\plantvillage_hf`
- Generated: 2026-10-25T19:15:23
- Images scanned: **54,305**
- Corrupt / unreadable: **0**
- Exact duplicate groups: **24** (24 redundant files)
- Near-duplicate pairs (approx., Hamming <= 6): **558**
- Pairs crossing a split boundary: **0** (near: 0, exact: 0)

## Variants

| variant | images | classes |
|---|---:|---:|
| color | 54,305 | 38 |

## Warnings

- None.

## Image dimensions

| width x height | count |
|---|---:|
| 256 x 256 | 54,305 |

## Class counts

| class | color |
|---|---:|
| Apple___Apple_scab | 630 |
| Apple___Black_rot | 621 |
| Apple___Cedar_apple_rust | 275 |
| Apple___healthy | 1,645 |
| Blueberry___healthy | 1,502 |
| Cherry_(including_sour)___Powdery_mildew | 1,052 |
| Cherry_(including_sour)___healthy | 854 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 513 |
| Corn_(maize)___Common_rust_ | 1,192 |
| Corn_(maize)___Northern_Leaf_Blight | 985 |
| Corn_(maize)___healthy | 1,162 |
| Grape___Black_rot | 1,180 |
| Grape___Esca_(Black_Measles) | 1,383 |
| Grape___Leaf_blight_(Isariopsis_Leaf_Spot) | 1,076 |
| Grape___healthy | 423 |
| Orange___Haunglongbing_(Citrus_greening) | 5,507 |
| Peach___Bacterial_spot | 2,297 |
| Peach___healthy | 360 |
| Pepper,_bell___Bacterial_spot | 997 |
| Pepper,_bell___healthy | 1,478 |
| Potato___Early_blight | 1,000 |
| Potato___Late_blight | 1,000 |
| Potato___healthy | 152 |
| Raspberry___healthy | 371 |
| Soybean___healthy | 5,090 |
| Squash___Powdery_mildew | 1,835 |
| Strawberry___Leaf_scorch | 1,109 |
| Strawberry___healthy | 456 |
| Tomato___Bacterial_spot | 2,127 |
| Tomato___Early_blight | 1,000 |
| Tomato___Late_blight | 1,909 |
| Tomato___Leaf_Mold | 952 |
| Tomato___Septoria_leaf_spot | 1,771 |
| Tomato___Spider_mites Two-spotted_spider_mite | 1,676 |
| Tomato___Target_Spot | 1,404 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 5,357 |
| Tomato___Tomato_mosaic_virus | 373 |
| Tomato___healthy | 1,591 |

## Near-duplicate examples (first 20)

- `raw\color\Peach___Bacterial_spot\9c23d3ff-f9c5-49ff-b6a2-2a2583476f84___Rutg._Bact.S 1123.JPG` ~ `raw\color\Peach___Bacterial_spot\16ff8ca9-82a3-48f2-a5e8-e698bef2d58a___Rutg._Bact.S 1122.JPG`
- `raw\color\Orange___Haunglongbing_(Citrus_greening)\1a3a5d93-5025-473f-9ac2-5f963e50c5f9___UF.Citrus_HLB_Lab 0692.JPG` ~ `raw\color\Tomato___Late_blight\89c134ee-41f7-46b7-b4dc-4229b3b3ca08___GHLB2 Leaf 8952.JPG`
- `raw\color\Orange___Haunglongbing_(Citrus_greening)\1a3a5d93-5025-473f-9ac2-5f963e50c5f9___UF.Citrus_HLB_Lab 0692.JPG` ~ `raw\color\Cherry_(including_sour)___Powdery_mildew\3a9dde01-97de-4650-b6e8-1f76b1597e7e___FREC_Pwd.M 4864.JPG`
- `raw\color\Orange___Haunglongbing_(Citrus_greening)\1a3a5d93-5025-473f-9ac2-5f963e50c5f9___UF.Citrus_HLB_Lab 0692.JPG` ~ `raw\color\Tomato___Septoria_leaf_spot\131407de-62dc-44bf-81c8-b0c4c173df9e___JR_Sept.L.S 2726.JPG`
- `raw\color\Tomato___Late_blight\89c134ee-41f7-46b7-b4dc-4229b3b3ca08___GHLB2 Leaf 8952.JPG` ~ `raw\color\Cherry_(including_sour)___Powdery_mildew\3a9dde01-97de-4650-b6e8-1f76b1597e7e___FREC_Pwd.M 4864.JPG`
- `raw\color\Tomato___Bacterial_spot\9e6a8241-00d3-474b-a664-1f67015d5187___UF.GRC_BS_Lab Leaf 0309.JPG` ~ `raw\color\Tomato___Tomato_Yellow_Leaf_Curl_Virus\1c1f329f-d726-463f-8e54-a5ed2e7c9e87___UF.GRC_YLCV_Lab 03254.JPG`
- `raw\color\Peach___Bacterial_spot\e785d770-55ee-4c53-b8cf-4ec12949fd83___Rut._Bact.S 3447.JPG` ~ `raw\color\Tomato___Leaf_Mold\10671d95-f882-4e19-994f-b35917d2c763___Crnl_L.Mold 9098.JPG`
- `raw\color\Tomato___Early_blight\a48e8b60-3636-4f7e-8ce2-d6625d866720___RS_Erly.B 7458.JPG` ~ `raw\color\Apple___Cedar_apple_rust\69ceb6f0-6e63-4676-90a6-288ae97b0c2f___FREC_C.Rust 0107.JPG`
- `raw\color\Peach___Bacterial_spot\33b013fd-0c6b-49e4-9784-36a68fbec3a2___Rutg._Bact.S 2171.JPG` ~ `raw\color\Peach___Bacterial_spot\63fd3fa5-29c6-472d-a61d-85d6ba93b04c___Rut._Bact.S 1485.JPG`
- `raw\color\Corn_(maize)___healthy\94ca39a0-d4df-44c6-b5cc-a38a754506e5___R.S_HL 0601 copy 3.jpg` ~ `raw\color\Corn_(maize)___healthy\8afc9635-c5e3-41be-aac0-74109d6112ed___R.S_HL 0601 copy 2.jpg`
- `raw\color\Tomato___Tomato_Yellow_Leaf_Curl_Virus\0aa5f107-5a75-4679-a316-634a840660fe___YLCV_NREC 2657.JPG` ~ `raw\color\Apple___healthy\230a4c38-8e59-4c1d-bda8-ee3401e6507b___RS_HL 7945.JPG`
- `raw\color\Tomato___Late_blight\5f21282c-e2ef-4c4a-ace1-b5701fe7effc___GHLB2 Leaf 8999.JPG` ~ `raw\color\Tomato___Late_blight\e5d707cd-077c-43af-bda9-6138e516ff51___GHLB2 Leaf 8999.JPG`
- `raw\color\Strawberry___Leaf_scorch\26677ce3-16e9-4035-9945-3392d0caeb61___RS_L.Scorch 1164.JPG` ~ `raw\color\Tomato___Tomato_Yellow_Leaf_Curl_Virus\321df6d9-e75b-45b7-bc40-133b248c58ed___YLCV_GCREC 2803.JPG`
- `raw\color\Peach___Bacterial_spot\bcc95b12-c380-4ece-92be-92ee6e1cff88___Rut._Bact.S 0788.JPG` ~ `raw\color\Blueberry___healthy\64a59e6a-476a-4eef-bfa3-5bc71d9b877e___RS_HL 2582.JPG`
- `raw\color\Blueberry___healthy\64a59e6a-476a-4eef-bfa3-5bc71d9b877e___RS_HL 2582.JPG` ~ `raw\color\Tomato___Tomato_Yellow_Leaf_Curl_Virus\5f632a9f-ce01-47e5-92d6-77e66a58bc45___UF.GRC_YLCV_Lab 09370.JPG`
- `raw\color\Peach___Bacterial_spot\d9dc64d7-43ff-40d9-93ff-fcec68cfc25f___Rut._Bact.S 0771.JPG` ~ `raw\color\Peach___Bacterial_spot\b2db309a-d37e-4093-a825-c119d3248c86___Rut._Bact.S 0785.JPG`
- `raw\color\Peach___Bacterial_spot\d9dc64d7-43ff-40d9-93ff-fcec68cfc25f___Rut._Bact.S 0771.JPG` ~ `raw\color\Peach___Bacterial_spot\485e0319-4f79-4504-b3eb-8c3f2dcc0c3e___Rut._Bact.S 0750.JPG`
- `raw\color\Peach___Bacterial_spot\d9dc64d7-43ff-40d9-93ff-fcec68cfc25f___Rut._Bact.S 0771.JPG` ~ `raw\color\Peach___Bacterial_spot\ef32ff5c-d480-4b68-941b-821ee2dbcf15___Rutg._Bact.S 2216.JPG`
- `raw\color\Peach___Bacterial_spot\ee63ad7f-a004-45ea-a6d7-2a859cf9ca03___Rut._Bact.S 1372.JPG` ~ `raw\color\Peach___Bacterial_spot\265e6c18-6d4c-4ea7-8677-da5d2ffe94ee___Rut._Bact.S 0860.JPG`
- `raw\color\Peach___Bacterial_spot\265e6c18-6d4c-4ea7-8677-da5d2ffe94ee___Rut._Bact.S 0860.JPG` ~ `raw\color\Peach___Bacterial_spot\b2db309a-d37e-4093-a825-c119d3248c86___Rut._Bact.S 0785.JPG`
