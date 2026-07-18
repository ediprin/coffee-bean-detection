# Coffee Bean Detection

Repo ini khusus untuk object detection biji kopi. Ia tidak mengimpor, mengubah,
atau memakai checkpoint repo klasifikasi Coffee-17. Baseline pertama adalah
YOLO26n; keluarga detector lain baru ditambahkan setelah baseline terkunci.

## Eksperimen VA-DCP

Pipeline offline `Visibility-Aware Dense Copy-Paste` sudah tersedia tanpa
mengubah internal YOLO:

```text
prepare_object_library
        -> profile_vadcp_source (prior empiris train nyata)
        -> generate_vadcp_dataset (empirical A1 atau visibility-aware A2)
        -> audit_vadcp
        -> audit_vadcp_realism + visual/cutout audit
        -> run_vadcp_ablation (A0/A1/A2)
```

Val dan test selalu berasal dari data nyata; generator hanya menambah train.
Metadata penuh menyimpan visible/full mask, z-order, visibility ratio, dan ID
aset sumber, sedangkan label YOLO menggunakan visible bounding box. Protokol dan
perintah lengkap ada di [docs/VA_DCP_IMPLEMENTATION.md](docs/VA_DCP_IMPLEMENTATION.md).
Generator memakai physics-informed 2.5D projected packing: skala dan jumlah
objek dikalibrasi dari train nyata, penempatan contact-constrained, z-order
eksplisit, dan cahaya/bayangan koheren per scene. Ini bukan klaim simulasi fisika
3D penuh.

Notebook Colab yang hanya menyiapkan data sampai status `TRAINING_READY`, tanpa
menjalankan training, tersedia di
[notebooks/VA_DCP_Setup_Colab.ipynb](notebooks/VA_DCP_Setup_Colab.ipynb).

## Baseline aktif

- `D0`: YOLO26n standar, tanpa HBP, attention, loss tambahan, atau modifikasi
  arsitektur.
- Baseline dilatih pada grouped split yang lolos audit dan dievaluasi sekali pada
  test split terkunci.
- Eksperimen lama `Y0/Y1` tetap disimpan untuk pekerjaan berikutnya, tetapi tidak
  dijalankan oleh runner baseline.

## Kontrak eksperimen lama

- `Y0`: YOLO11n standar.
- `Y1`: YOLO11n dengan local bilinear adapter hanya pada classification branch
  `Detect.cv3`; box regression `Detect.cv2` tidak diubah.
- Dataset Coffee Defect v11 diperlakukan sebagai dataset roasted-bean enam kelas
  yang independen. Labelnya tidak dipetakan ke Coffee-17 tanpa audit visual.
- Dataset dan checkpoint tidak disimpan di Git.

## Kaggle: instalasi dan audit

```python
%cd /kaggle/working/coffee-bean-detection
!pip install -q -e .

from pathlib import Path

matches = list(Path("/kaggle/input").rglob("data.yaml"))
print("data.yaml ditemukan:")
for path in matches:
    print("-", path)
```

Pilih folder induk `data.yaml`, kemudian audit:

```python
import subprocess, sys

RAW_ROOT = "/kaggle/input/NAMA-DATASET/FOLDER-DATASET"
subprocess.run([
    sys.executable, "-u", "-m", "coffee_detector.audit_dataset",
    "--data-root", RAW_ROOT,
    "--output", "/kaggle/working/coffee-defect-audit.json",
], check=True)
```

Jika `AMAN TRAINING: BELUM`, buat grouped split baru. Folder output harus baru
atau kosong:

```python
subprocess.run([
    sys.executable, "-u", "-m", "coffee_detector.prepare_dataset",
    "--data-root", RAW_ROOT,
    "--output-root", "/kaggle/working/coffee-defect-v11-clean",
    "--seed", "42",
], check=True)
```

## Screening hemat Y0 vs Y1

Jalankan satu seed terlebih dahulu. Runner memakai progress bawaan Ultralytics,
melewati `best.pt` yang sudah ada, dan melanjutkan `last.pt` setelah sesi putus.

```python
DATA_ROOT = "/kaggle/working/coffee-defect-v11-clean"
OUTPUT_ROOT = "/kaggle/working/coffee-yolo-results"

subprocess.run([
    sys.executable, "-u", "-m", "coffee_detector.run_screening",
    "--data-root", DATA_ROOT,
    "--output-root", OUTPUT_ROOT,
    "--seed", "42",
    "--device", "0",
], check=True)
```

Jika audit sudah dijalankan terpisah, hindari pemindaian ulang dengan memberikan
audit JSON yang telah menyatakan dataset aman:

```text
--verified-audit /path/to/coffee-defect-raw-audit.json
```

Runner memverifikasi bahwa `dataset_root` dalam JSON sama dengan `--data-root`;
audit dari dataset lain akan ditolak.

`best.pt` dapat muncul sebelum training selesai. Runner hanya melewati training
bila `best.pt` dan `experiment_manifest.json` sama-sama ada. Run yang terputus
akan dilanjutkan dari `last.pt`, bukan salah dianggap selesai.

Hasil utama disimpan di:

```text
/kaggle/working/coffee-yolo-results/
├── Y0_seed42/weights/best.pt
├── Y1_seed42/weights/best.pt
└── reports/screening_seed42.json
```

Jangan menjalankan seed 123 dan 2026 sebelum Y1 menunjukkan peningkatan pada
`mAP50-95` dan worst-class AP pada seed 42.

Versi Ultralytics dipatok ke `8.4.96`. Versi ini memuat definisi YOLO26 dan juga
menjaga kontrak internal eksperimen lama `Detect.cv2/cv3`. Jangan memperbaruinya
tanpa menjalankan ulang seluruh tes.

## Kaggle: baseline YOLO26n saja

Setelah dataset Roboflow ditambahkan sebagai Kaggle Input, jalankan satu sel ini.
Sel akan mencari `data.yaml`, mengaudit data, membuat grouped split bila ada
duplikasi lintas split, lalu melatih dan mengevaluasi `D0` pada test split.

Grouped split memakai exact image hash dan parent ID Roboflow. Kemiripan visual
berbasis perceptual hash hanya dilaporkan sebagai peringatan karena gambar biji
kopi yang homogen dapat membentuk false-positive chain bila dipakai sebagai grup.

```python
%cd /kaggle/working/coffee-bean-detection

from pathlib import Path
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)

from coffee_detector.audit_dataset import audit_dataset
from coffee_detector.prepare_dataset import prepare_dataset

candidates = [
    path.parent
    for path in Path("/kaggle/input").rglob("data.yaml")
    if "coffee" in str(path).lower()
]
print("Kandidat dataset:")
for path in candidates:
    print("-", path)
assert len(candidates) == 1, "Pilih RAW_ROOT secara manual dari daftar di atas."

RAW_ROOT = candidates[0]
PREPARED_ROOT = Path("/kaggle/working/coffee-defect-v11-clean")
audit = audit_dataset(RAW_ROOT, "/kaggle/working/coffee-defect-raw-audit.json")

if audit["safe_for_training"]:
    DATA_ROOT = RAW_ROOT
else:
    if not (PREPARED_ROOT / "data.yaml").is_file():
        prepare_dataset(RAW_ROOT, PREPARED_ROOT, seed=42)
    DATA_ROOT = PREPARED_ROOT

subprocess.run([
    sys.executable, "-u", "-m", "coffee_detector.run_baseline",
    "--data-root", str(DATA_ROOT),
    "--output-root", "/kaggle/working/yolo26-baseline-results",
    "--seed", "42",
    "--device", "0",
], check=True)
```

Output yang perlu disimpan:

```text
/kaggle/working/yolo26-baseline-results/
├── D0_seed42/weights/best.pt
└── reports/
    ├── dataset_audit.json
    ├── D0_seed42_test.json
    └── D0_seed42_summary.json
```

## Audit visual baseline

Setelah test selesai, audit seluruh gambar test dan buat pasangan ground truth
versus prediksi untuk kasus dengan count error atau confidence terendah:

```bash
python -u -m coffee_detector.run_visual_audit \
  --checkpoint /path/to/D0_seed42/weights/best.pt \
  --data-root /path/to/dataset \
  --output-root /path/to/visual-audit \
  --samples 12 \
  --batch-size 16 \
  --device 0
```

Runner menulis `visual_audit.json`, gambar individual di `pairs/`, dan
`contact_sheet.jpg`. Kotak ground truth berwarna biru dan prediksi merah.
