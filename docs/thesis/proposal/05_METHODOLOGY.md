# BAB III — METODE PENELITIAN

Status: **proposal draft setelah penyesuaian terhadap desain optimasi AF2, audit metodologi Hong et al., protokol direct-from-pretrained, dan kontrak dataset Faruq-v3**.

Bab ini membedakan dengan tegas empat lapisan penelitian:

1. **pengembangan/optimasi struktur AF2** melalui analisis faktor satu-per-satu;
2. **pemilihan konfigurasi AF2** tanpa menggunakan locked test;
3. **eksperimen konfirmatori** native YOLO26n versus AF2-YOLO26n dengan kontrol yang dipasangkan;
4. **analisis hasil** yang mencakup performa agregat, lower-tail classes, diagnosis mekanisme, visualisasi, kesalahan, dan efisiensi.

Sumber teknis utama bab ini adalah:

- `docs/thesis/foundation/08_BAB3_HONG_ADAPTED_OPTIMIZATION_DESIGN.md`;
- `docs/FARUQ_V3_AF2_SPECTRAL_FACTORIZATION_PROTOCOL.md`;
- `docs/FARUQ_V3_AF2_DIRECT_FROM_PRETRAINED_PROTOCOL_2026-08-24.md`;
- `configs/afab/AF2_yolo26n_chaotic_amplitude.yaml`;
- `src/coffee_detector/afab/operator.py`;
- kontrak grouped split Faruq-v3 yang telah diaudit pada repository.

Hong et al. [COF-01] digunakan sebagai **template metodologis** untuk pola overall architecture → systematic ablation/sensitivity → quantitative evaluation → visualization/error analysis. Modul DSConv, SPPF-Attention, PConv, skema 5-fold, dan hyperparameter YOLOv10 milik Hong **tidak disalin** ke metode penelitian ini.

---

## 3.1 Kerangka Penelitian

Penelitian ini menguji apakah preprocessing citra berbasis frekuensi-angular dapat meningkatkan kemampuan YOLO26 dalam mendeteksi cacat biji kopi yang bersifat fine-grained. Metode ditempatkan pada ruang input sehingga arsitektur internal detector tidak diubah.

Secara umum, penelitian disusun melalui alur berikut.

```text
Kajian masalah fine-grained coffee defect
        ↓
Audit dataset dan grouped split
        ↓
Definisi AF2 reference
        ↓
Analisis faktor / kandidat struktur AF2
        ↓
Sensitivity parameter terpilih
        ↓
Pemilihan AF2* dan method freeze
        ↓
Matched confirmatory experiment
Native YOLO26n  vs  AF2*-YOLO26n
        ↓
Evaluasi aggregate + lower tail
        ↓
Mechanism diagnostics
        ↓
Visualization + error analysis
        ↓
Efficiency analysis
        ↓
Kesimpulan penelitian
```

Filosofi eksperimen mengikuti prinsip bahwa sebuah perubahan metode harus dapat dihubungkan dengan treatment yang jelas. Oleh karena itu, modifikasi AF2 tidak ditumpuk secara bebas. Setiap kandidat struktural mengubah satu keputusan desain dan dibandingkan terhadap AF2 reference yang sama.

**Gambar 3.1 pada dokumen final:** diagram kerangka penelitian dari kajian masalah sampai analisis akhir sebagaimana alur di atas.

---

## 3.2 Dataset Penelitian

### 3.2.1 Sumber dan karakteristik dataset

Penelitian menggunakan **Faruq-v3 grouped development archive** untuk object detection 21 kelas. Development split yang dibekukan terdiri atas train dan validation; seluruh 21 kelas target terdapat pada kedua split.

### Tabel 3.1 Ringkasan dataset pengembangan

| Split | Jumlah citra | Jumlah anotasi | Jumlah kelas | Penggunaan |
|---|---:|---:|---:|---|
| Train | 1.665 | 2.986 | 21 | Optimisasi parameter model pada training |
| Validation | 294 | 526 | 21 | Candidate screening, model selection, evaluasi dan diagnostic |
| Locked test | Tidak dibuka selama development | — | — | Evaluasi akhir setelah method freeze jika protokol final mengizinkan |

Label kelas mengikuti taxonomy Faruq-v3. Konteks SNI digunakan untuk menjelaskan relevansi cacat fisik, tetapi sistem tidak diasumsikan merekonstruksi keseluruhan prosedur grading, pembobotan nilai cacat, atau keputusan mutu SNI.

### 3.2.2 Taxonomy kelas

Unit prediksi detector adalah objek biji/cacat dengan bounding box dan kelas target. Karena penelitian berfokus pada fine-grained detection, evaluasi tidak hanya dilakukan pada nilai agregat tetapi juga pada AP setiap kelas serta kelompok kelas berperforma rendah.

Daftar nama 21 kelas akan ditempatkan pada lampiran atau tabel dataset final agar tidak memenuhi alur metodologi utama.

### 3.2.3 Grouped split dan kontrol kebocoran

Dataset menggunakan grouped split yang telah diaudit terhadap kebocoran antar-split. Kontrak data mensyaratkan:

1. tidak ada parent overlap antara train dan validation;
2. tidak ada exact-hash overlap antar-split;
3. development root yang digunakan selama screening tidak menyediakan direktori `test`;
4. test tidak digunakan untuk memilih kandidat AF2, threshold optimasi, atau hyperparameter preprocessing.

Grouped split dipertahankan sebagai bagian dari experimental contract dan tidak berubah berdasarkan hasil suatu kandidat.

### 3.2.4 Augmentasi dan transformasi input

Augmentasi training dari pipeline YOLO diperlakukan sebagai bagian dari training configuration dan harus identik pada arm yang dibandingkan. AF2 tidak diperlakukan sebagai augmentasi karena tidak menambah sampel baru dan tidak mengubah label bounding box melalui crop, translasi, atau warp koordinat.

Operator AF2 mempertahankan tinggi dan lebar tensor input. Meskipun demikian, preservation ukuran/geometri input tidak boleh ditafsirkan sebagai jaminan bahwa box prediction setelah training akan identik.

---

## 3.3 Baseline YOLO26

### 3.3.1 Arsitektur baseline

Detector utama adalah YOLO26n P3–P5 [DET-01]. Pada penelitian ini YOLO26 berfungsi sebagai **fixed detector family** untuk mengisolasi pengaruh preprocessing, bukan sebagai objek modifikasi backbone, neck, atau detection head.

Baseline didefinisikan sebagai:

\[
I \xrightarrow{\mathrm{YOLO26n}} \hat{Y}_{N}.
\]

Keluaran \(\hat{Y}_{N}\) mencakup prediksi bounding box, confidence, dan kelas sesuai mekanisme detector.

### 3.3.2 Pretrained initialization dan matched head

Eksperimen konfirmatori menggunakan official pretrained artifact yang sama pada native dan treatment. Frozen direct protocol mencatat:

```text
filename       = yolo26n.pt
source release = ultralytics/assets v8.4.0
source classes = 80
SHA-256        = 9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef
```

Karena source head berjumlah 80 kelas sedangkan Faruq-v3 memiliki 21 kelas, tensor head yang shape-incompatible memerlukan fresh initialization. Untuk mencegah random initialization menjadi confounder, native dan AF2 arm membuat target detector dalam isolated RNG fork dengan seed yang sama sebelum common pretrained weights dimuat.

Preflight pada eksperimen konfirmatori mensyaratkan seluruh persistent detector state native dan AF2 identik sebelum training dimulai.

---

## 3.4 Arsitektur Metode yang Diusulkan

Metode yang diusulkan menambahkan satu input frontend sebelum YOLO26n:

\[
I \xrightarrow{\mathrm{AF2}} I'
\xrightarrow{\mathrm{YOLO26n}} \hat{Y}_{AF2}.
\]

Dengan demikian, perbedaan arsitektur konseptual adalah:

```text
Native:
RGB image ─────────────────────────→ YOLO26n → prediction

Proposed:
RGB image → AF2 frequency-angular → YOLO26n → prediction
```

AF2 tidak mempunyai learned parameters dan tidak menjadi bagian dari backbone, neck, maupun head. Namun AF2 tetap melakukan operasi patch extraction, FFT, angular aggregation, inverse FFT, dan reconstruction sehingga tidak diasumsikan bebas biaya komputasi.

**Gambar 3.2 pada dokumen final:** diagram dua jalur paralel native dan proposed, dengan posisi AF2 sebelum detector ditandai secara eksplisit.

---

## 3.5 Preprocessing Frekuensi-Angular AF2

AF2 merupakan adaptasi input-space dari mekanisme angular frequency selection yang paling dekat dengan AFAB-2 pada LFDet [FG-01]. Penelitian ini membedakan bagian yang berasal dari prinsip parent method dan keputusan transfer implementasi repository seperti independent RGB processing, discretized angular bins, overlap averaging, dan residual image gate.

### 3.5.1 Pembentukan patch lokal

Untuk input:

\[
I \in \mathbb{R}^{B\times 3\times H\times W},
\]

AF2 reference membagi citra menjadi patch berukuran:

\[
m\times m,\qquad m=32.
\]

Dengan overlap \(o=0,50\), stride reference adalah:

\[
s=m(1-o)=16.
\]

Jika ukuran citra tidak tepat terhadap grid patch, padding `replicate` digunakan. Pada reference AF2, channel RGB diproses secara independen.

### 3.5.2 Transformasi Fourier

Untuk patch \(P_i\), transformasi Fourier ditulis:

\[
F_i=\operatorname{fftshift}\left(\mathcal{F}(P_i)\right).
\]

Magnitude dan phase masing-masing:

\[
A_i(u,v)=|F_i(u,v)|,
\]

\[
\phi_i(u,v)=\arg F_i(u,v).
\]

Bagian FFT dijalankan dalam float32 pada implementasi repository untuk keamanan CUDA/AMP, kemudian hasil dikembalikan ke dtype input.

### 3.5.3 Angular spectral density

Setiap koordinat frekuensi dipetakan ke arah:

\[
\theta(u,v)=\operatorname{atan2}(v-v_c,u-u_c),
\]

kemudian ke domain \([0,360^\circ)\). Pada AF2 reference digunakan \(K=360\) angular bins dengan floor-to-bin implementation.

Untuk channel \(c\), density arah \(k\) adalah:

\[
D_i^c(k)=
\sum_{(u,v):b(u,v)=k} A_i^c(u,v).
\]

Probabilitas angular:

\[
p_i^c(k)=
\frac{D_i^c(k)}{\sum_jD_i^c(j)+\varepsilon}.
\]

### 3.5.4 Entropy-adaptive threshold

Entropy angular dihitung sebagai:

\[
H_i^c=-\sum_k p_i^c(k)\log\left(p_i^c(k)+\varepsilon\right).
\]

Threshold adaptif:

\[
\tau_i^c=
\frac{\gamma}{1+\exp(-H_i^c)},
\qquad \gamma=0,10
\]

pada konfigurasi reference. Karena \(H_i^c\) berasal dari patch/channel yang sedang diproses, threshold berubah terhadap content walaupun tidak dipelajari sebagai trainable parameter.

### 3.5.5 Directional weighting

Density dinormalisasi terhadap nilai maksimum:

\[
q_i^c(k)=
\frac{D_i^c(k)}{\max_jD_i^c(j)+\varepsilon}.
\]

Pada reference AF2 digunakan hard threshold:

\[
w_i^c(k)=
\begin{cases}
0, & q_i^c(k)\le \tau_i^c,\\
q_i^c(k), & q_i^c(k)>\tau_i^c.
\end{cases}
\]

Koefisien Fourier kemudian dibobotkan:

\[
\widetilde F_i^c(u,v)=
F_i^c(u,v)\,w_i^c(b(u,v)).
\]

### 3.5.6 Inverse FFT dan spatial reconstruction

Patch difilter kemudian dikembalikan ke ruang spasial:

\[
\widetilde P_i=
\Re\left\{
\mathcal{F}^{-1}
\left(\operatorname{ifftshift}(\widetilde F_i)\right)
\right\}.
\]

Patch overlap digabungkan menggunakan fold/overlap averaging sehingga diperoleh:

\[
R_{AF2}(I).
\]

### 3.5.7 Residual image enhancement

Respons spasial dinormalisasi:

\[
G(I)=\operatorname{MinMax}(R_{AF2}(I)),
\]

kemudian digabungkan dengan citra asli:

\[
\boxed{
I'=I+I\odot G(I)
}
\]

sehingga shape input dipertahankan:

\[
\operatorname{shape}(I')=\operatorname{shape}(I).
\]

### Tabel 3.2 Konfigurasi AF2 reference

| Parameter | Nilai | Peran |
|---|---:|---|
| mode | `af2` | angular-density filtering |
| patch size | 32 | skala spasial analisis lokal |
| overlap | 0,50 | overlap antarpatch |
| gamma | 0,10 | skala entropy-conditioned threshold |
| angular bins | 360 | resolusi diskret arah |
| chunk size | 128 | engineering/memory control |
| epsilon | \(10^{-8}\) | stabilitas numerik |
| channel processing | independent RGB | keputusan transfer implementation |
| reconstruction | fold + overlap average | penggabungan patch |
| learned parameter | 0 | parameter-free frontend |

`radius_ratio=0.05` terdapat pada konfigurasi bersama AFAB, tetapi hanya aktif pada jalur AF1/AF12 radial mask dan **tidak menjadi parameter AF2 mode** pada penelitian ini.

**Gambar 3.3 pada dokumen final:** `RGB → patch → FFT → angular density → entropy threshold → weighted spectrum → IFFT → overlap reconstruction → residual enhancement`.

---

## 3.6 Analisis dan Optimasi AF2

Bagian ini mengoperasionalkan kata **Optimasi** pada judul penelitian. Strategi diadaptasi dari pola systematic ablation dan sensitivity analysis Hong et al. [COF-01], tetapi diterapkan pada keputusan desain AF2 secara satu-faktor-pada-satu-waktu.

Optimasi tidak berarti menambahkan modul secara progresif. Tujuannya adalah mengidentifikasi konfigurasi preprocessing yang memberikan trade-off terbaik antara performa agregat, lower-tail classes, dan biaya komputasi.

### 3.6.1 Factorization of AF2 design

AF2 reference mengandung beberapa keputusan desain yang dapat memengaruhi hasil: rectangular window, representasi 360 arah, tidak adanya radial factorization, hard threshold, dan independent RGB gates. Kandidat berikut mengubah **satu faktor** terhadap reference.

### Tabel 3.3 Kandidat optimasi struktural AF2

| Kandidat | Perubahan tunggal | Pertanyaan ilmiah |
|---|---|---|
| `AF2C` | AF2 reference | kontrol struktur |
| `AF2WIN` | rectangular → square-root Hann + normalized overlap-add | apakah windowing mengurangi spectral leakage secara berguna? |
| `AF2ORI` | 360 direction bins → 16 orientation bins modulo \(\pi\) | apakah orientation representation yang lebih ringkas lebih sesuai? |
| `AF2POL` | angular-only → 3 radial bands × 16 orientations | apakah struktur radial memberikan informasi tambahan? |
| `AF2SOFT` | hard threshold → soft weighting | apakah selection yang kontinu lebih stabil? |
| `AF2LUM` | independent RGB → Rec.709 luminance shared gate | apakah pemisahan channel RGB diperlukan? |

Kandidat tersebut **bukan module stack**. Jika satu kandidat gagal, kegagalan tersebut tetap dilaporkan sebagai evidence terhadap keputusan desain yang diuji.

`PCG1` dan `WAV1` pada genealogy repository diperlakukan hanya sebagai optional mechanistic comparator karena keduanya bukan AF2 variants. Keduanya tidak menentukan konfigurasi optimal AF2.

### 3.6.2 Structural candidate screening

Screening struktural dilakukan pada development train/validation tanpa akses ke locked test. Pada protokol factorization yang telah dibekukan, semua candidate arm memakai schedule yang sama dan seed 42. Rule historis repository untuk mempertahankan kandidat terhadap `AF2C` adalah:

1. Macro mAP50–95 meningkat sedikitnya 0,5 percentage point;
2. Bottom-3 mAP50–95 tidak menurun;
3. Worst-class mAP50–95 tidak turun lebih dari 1 percentage point;
4. validation memuat seluruh 21 ground-truth classes;
5. test tidak diakses.

Jika lebih dari satu kandidat lolos, prioritas utama adalah Macro mAP50–95. Kandidat dengan selisih Macro kurang dari 0,2 point dibandingkan kandidat teratas dibedakan berturut-turut menggunakan Bottom-3, Worst-class, kemudian batch-1 latency.

Perlu dibedakan dua source-of-evidence:

- genealogy factorization lama menggunakan seed-matched **D0 coffee checkpoint** sebagai parent untuk candidate screening;
- eksperimen konfirmatori utama tesis menggunakan **official YOLO26n pretrained checkpoint langsung**.

Karena initial state kedua tahap berbeda, hasil candidate screening lama diperlakukan sebagai **development/selection evidence**, bukan sebagai final proof bahwa kandidat yang sama pasti unggul pada direct-from-pretrained setting. Konfigurasi final tetap harus melewati eksperimen konfirmatori yang dijelaskan pada §3.7.

### 3.6.3 Parameter sensitivity analysis

Setelah struktur AF2 dipilih, parameter yang secara metodologis dapat dianalisis adalah:

\[
\Theta_{AF2}=\{m,o,\gamma,K\},
\]

Dengan:

- \(m\): patch size;
- \(o\): overlap;
- \(\gamma\): entropy-threshold coefficient;
- \(K\): angular-bin resolution.

Untuk menjaga ruang lingkup tesis, sensitivity analysis final tidak wajib menyapu semua parameter. Prioritas utama adalah \(\gamma\) dan \(m\), karena keduanya secara langsung mengubah kekuatan angular selection dan skala spasial analisis lokal.

Exact candidate values untuk sensitivity tambahan **belum didukung oleh frozen source saat ini** dan tidak ditentukan secara retrospektif dari hasil validation. Jika tahap ini dijalankan, nilai kandidat harus dibekukan dalam protocol amendment sebelum hasil corresponding run diamati.

`chunk_size` dan `eps` tidak diperlakukan sebagai scientific optimization variables. `radius_ratio` juga tidak dipakai karena tidak aktif pada mode AF2.

### 3.6.4 Configuration selection dan method freeze

Alur selection adalah:

```text
AF2 reference
    ↓
factorized structural screening
    ↓
limited parameter sensitivity jika diperlukan
    ↓
AF2* selected
    ↓
METHOD FREEZE
    ↓
confirmatory experiment
```

Tidak ada perubahan konfigurasi AF2 setelah locked test dibuka. Dengan demikian, test tidak berfungsi sebagai hyperparameter-selection oracle.

---

## 3.7 Rancangan Eksperimen Konfirmatori

### 3.7.1 Native versus AF2*

Eksperimen konfirmatori membandingkan:

\[
I \xrightarrow{\mathrm{YOLO26n}} \hat{Y}_{N}
\]

melawan:

\[
I \xrightarrow{\mathrm{AF2^*}} I'
\xrightarrow{\mathrm{YOLO26n}} \hat{Y}_{A}.
\]

### Tabel 3.4 Arm eksperimen konfirmatori

| Arm | Initialization | Input treatment | Detector | Fungsi |
|---|---|---|---|---|
| Native | official `yolo26n.pt` + matched 21-class head | none | YOLO26n P3–P5 | kontrol |
| AF2* | exact same source/state | selected AF2 aktif sejak forward pertama | YOLO26n P3–P5 | treatment |

Paired delta untuk metric \(M\):

\[
\Delta M=M_{AF2^*}-M_{Native}.
\]

Seed 42 direct-AF2 yang sudah tersedia diperlakukan sebagai feasibility pilot. Hasil pilot tidak digunakan sebagai pengganti konfirmasi multi-seed.

### 3.7.2 Repeated paired seeds

Konfirmasi direncanakan pada:

\[
\mathcal S=\{42,123,2026\}.
\]

Untuk setiap seed, native dan AF2 arm menggunakan paired initialization dan common training protocol. Hasil dilaporkan sebagai nilai per seed, mean antar-seed, dan paired deltas. Jika arah perubahan tidak stabil antar-seed, ketidakstabilan tersebut dilaporkan sebagai temuan.

### 3.7.3 Preflight fairness gates

Sebelum training confirmatory run, harus dipastikan:

1. SHA pretrained source identik;
2. model YAML identik;
3. training schedule identik;
4. selected AF2 mapping sesuai method freeze;
5. target-head initialization dipadankan;
6. persistent detector states identik sebelum training;
7. detector parameter count identik;
8. AF2 learned parameter count = 0;
9. AF2 probe finite, aktif, dan shape-preserving;
10. dataset audit lulus;
11. locked test tidak digunakan selama selection.

### 3.7.4 Locked-test protocol

Locked test, jika digunakan pada tahap tesis final, hanya dibuka setelah:

- structural/parameter selection selesai;
- AF2* dibekukan;
- decision rule konfirmasi sudah ditetapkan;
- tidak ada rencana mengubah konfigurasi berdasarkan test outcome.

---

## 3.8 Konfigurasi Pelatihan

Konfigurasi confirmatory training mengikuti frozen direct-from-pretrained protocol.

### Tabel 3.5 Konfigurasi pelatihan

| Parameter | Nilai |
|---|---:|
| maximum epoch | 50 |
| image size | 640 |
| batch size | 16 |
| workers | 2 |
| patience | 15 |
| optimizer | `auto` |
| pretrained | `true` |
| cache | `false` |
| close mosaic | 10 |
| maximum detections | 500 |
| deterministic | `true` |
| paired seeds | 42, 123, 2026 |

Early stopping mengikuti `patience=15`; 50 epoch merupakan maksimum dan tidak berarti kedua arm harus berhenti pada epoch yang sama.

Repository memiliki sejumlah schedule dari eksperimen lama. Untuk final direct confirmatory experiment, frozen direct-from-pretrained protocol adalah authority dan schedule lama tidak digunakan untuk mendefinisikan arm utama.

---

## 3.9 Metrik Evaluasi

Evaluasi dibagi menjadi performance aggregate, lower-tail performance, dan per-class behavior. Metrik standar detector dilengkapi study-defined tail metrics agar performa kelas sulit tidak tertutup oleh nilai rata-rata.

### 3.9.1 Precision, Recall, dan F1

Untuk kebutuhan descriptive error analysis:

\[
P=\frac{TP}{TP+FP},
\]

\[
R=\frac{TP}{TP+FN},
\]

\[
F1=2\frac{PR}{P+R}.
\]

Precision/Recall/F1 bukan primary selection metric AF2 tetapi dapat digunakan untuk menjelaskan pola kelas atau error tertentu.

### 3.9.2 mAP50 dan mAP50–95

mAP50 digunakan sebagai secondary context, sedangkan mAP50–95 digunakan karena mengevaluasi AP pada beberapa threshold IoU dari 0,50 sampai 0,95. Final calculation mengikuti implementation evaluator yang dibekukan pada repository/Ultralytics dan akan dirujuk ke standar evaluasi yang sesuai pada dokumen akhir [EVAL-01][EVAL-02].

### 3.9.3 Macro mAP50–95

Jika \(AP_c\) adalah AP kelas \(c\) pada IoU 0,50–0,95 dan terdapat \(C=21\) kelas:

\[
\mathrm{Macro}=
\frac{1}{C}\sum_{c=1}^{C}AP_c.
\]

Macro mAP50–95 menjadi primary aggregate metric penelitian.

### 3.9.4 Bottom-3 mAP50–95

Urutkan AP kelas:

\[
AP_{(1)}\le AP_{(2)}\le\dots\le AP_{(C)}.
\]

Bottom-3:

\[
\mathrm{Bottom3}=
\frac{AP_{(1)}+AP_{(2)}+AP_{(3)}}{3}.
\]

### 3.9.5 Worst-class dan per-class AP

Worst-class:

\[
\mathrm{Worst}=AP_{(1)}.
\]

Bottom-3 dan Worst-class adalah **study-defined summary metrics**, bukan official COCO metrics. Per-class AP tetap dilaporkan agar perubahan pada masing-masing kategori dapat diperiksa langsung.

### Tabel 3.6 Hierarki metrik performa

| Tingkat | Metrik | Fungsi |
|---|---|---|
| Aggregate | Macro mAP50–95 | primary overall comparison |
| Secondary aggregate | mAP50, mAP50–95 | detector context |
| Tail | Bottom-3 | kestabilan tiga kelas terendah |
| Safety indicator | Worst-class | kelas paling lemah |
| Detailed | per-class AP | identifikasi gain/regression per kategori |

---

## 3.10 Analisis Mekanisme

Analisis mekanisme bertujuan membedakan pola yang berkaitan dengan proposal/localization accessibility dari pola yang lebih konsisten dengan class discrimination. Diagnosis dilakukan pada validation setelah best checkpoint tersedia.

Frozen direct diagnostic menggunakan:

```text
imgsz        = 640
match IoU    = 0.50
confidence   = 0.25
NMS IoU      = 0.70
max_det      = 500
raw counts   = 50, 100, 300, 500
```

### Tabel 3.7 Metrik diagnostic

| Diagnostic | Interpretasi yang diizinkan |
|---|---|
| Raw top-500 proposal accessibility | availability target pada raw/top proposals; proxy terbatas proposal/localization accessibility |
| Localization-conditioned Top-1 | class decision setelah syarat lokalisasi dipenuhi; proxy diskriminasi kelas |
| Correct-decision recall | proporsi keputusan benar pada target evaluasi |

Diagnostic tidak menggantikan mAP dan tidak membuktikan kausalitas. Jika raw proposal accessibility relatif tetap tetapi localization-conditioned Top-1 meningkat, kesimpulan dibatasi menjadi:

> pola hasil **lebih konsisten dengan** peningkatan diskriminasi kelas daripada peningkatan raw proposal accessibility.

Penelitian tidak menyatakan bahwa box-regression quality secara keseluruhan pasti tidak berubah hanya berdasarkan diagnostic ini.

---

## 3.11 Analisis Visualisasi

Hong et al. menggunakan activation visualization sebagai dukungan interpretasi. Penelitian ini mengadaptasi prinsip tersebut, tetapi karena treatment berada di input-space, visualisasi difokuskan terlebih dahulu pada transformasi AF2 itu sendiri.

### 3.11.1 Visualisasi transformasi AF2

Untuk selected samples, panel visual yang direncanakan meliputi:

```text
Original RGB
→ selected patch
→ FFT magnitude
→ angular density D(θ)
→ adaptive threshold τ
→ retained angular response
→ reconstructed spatial cue
→ AF2-enhanced RGB
→ prediction
```

Visualisasi ini digunakan untuk menunjukkan apa yang dilakukan operator, bukan untuk membuktikan bahwa suatu pola spektral merupakan penyebab keputusan model.

### 3.11.2 Activation visualization

CAM/EigenCAM atau teknik visualization detector lain dapat digunakan hanya setelah kompatibilitasnya dengan YOLO26 diverifikasi. Oleh karena itu proposal tidak mengunci nama metode CAM tertentu sebelum audit teknis tersebut selesai.

Jika digunakan, native dan AF2 harus memakai layer target, normalization, image identity, dan rendering protocol yang sama.

### 3.11.3 Pemilihan contoh visual

Untuk mengurangi cherry-picking, qualitative panels tidak dipilih semata-mata berdasarkan contoh yang tampak paling menarik. Sampel akan diambil dari outcome groups yang sudah didefinisikan pada §3.12 menggunakan aturan deterministic/fixed-seed sampling yang dibekukan sebelum visual inspection final.

---

## 3.12 Analisis Kesalahan

Analisis kesalahan mengadaptasi confusion-matrix approach Hong dan memperluasnya menjadi paired native-versus-AF2 transition analysis.

### 3.12.1 Confusion dan per-class error

Untuk matched validation identities, penelitian memeriksa:

- per-class AP;
- false positive dan false negative per kelas sejauh tersedia;
- pasangan kelas yang sering tertukar;
- kelas yang berulang kali berada pada Bottom-3;
- kelas dengan gain/loss terbesar;
- konsistensi error lintas seed.

Confusion matrix harus menggunakan matching/evaluation protocol yang didefinisikan dan tidak boleh dicampur dengan metrik classification-only tanpa penjelasan.

### 3.12.2 Rescue–regression transition

Setiap matched target dikategorikan menjadi:

| Native | AF2 | Kategori |
|---|---|---|
| salah | benar | AF2 rescue |
| benar | salah | AF2 regression |
| salah | salah | unresolved |
| benar | benar | stable correct |

Untuk kelas \(c\):

\[
R_c=N(\text{native wrong, AF2 correct}),
\]

\[
G_c=N(\text{native correct, AF2 wrong}),
\]

serta statistik deskriptif:

\[
NR_c=R_c-G_c.
\]

`NR_c` adalah statistik analisis yang dirancang untuk tesis ini dan **bukan** diklaim sebagai metric standar dari Hong atau COCO.

Panel qualitative diprioritaskan pada empat kelompok: rescue, regression, unresolved, dan difficult lower-tail classes.

---

## 3.13 Evaluasi Efisiensi

AF2 tidak menambah learned parameters pada detector, tetapi menambah komputasi preprocessing. Evaluasi efisiensi mencakup:

1. detector parameter count;
2. end-to-end latency per image;
3. throughput;
4. peak GPU memory/VRAM;
5. kondisi runtime/hardware yang sama;
6. image size dan precision mode yang sama;
7. warm-up dan measurement procedure yang sama.

Perbandingan parameter saja tidak cukup untuk menyebut metode lightweight. Istilah **parameter-free** tidak disamakan dengan **compute-free**.

---

## 3.14 Lingkungan Implementasi dan Reproducibility

### Tabel 3.8 Informasi lingkungan yang wajib direkam

| Komponen | Rekaman |
|---|---|
| repository | branch + commit SHA |
| Python | versi runtime |
| PyTorch | versi |
| CUDA | versi |
| Ultralytics | versi; direct screen dibekukan pada 8.4.96 |
| GPU | model dan VRAM |
| pretrained source | filename + SHA-256 |
| random seed | nilai setiap run |
| dataset contract | split/version/leakage audit |
| run contract | arm, seed, config hash, checkpoint source, test lock |

Perbandingan latency dan memory hanya dianggap comparable jika hardware, runtime, image size, precision, warm-up, dan measurement procedure sama.

---

## 3.15 Batas Studi Pendahuluan, Optimasi, dan Bukti Final

Penelitian membedakan tiga jenis evidence.

### 3.15.1 Studi pendahuluan direct-AF2

Seed-42 direct experiment yang sudah tersedia hanya menunjukkan feasibility awal dan keputusan promosi ke evaluasi lebih lanjut. Hasil tersebut tidak diperlakukan sebagai final superiority claim.

### 3.15.2 Genealogy optimasi AF2

Eksperimen factorization yang telah dilakukan pada repository digunakan sebagai evidence pengembangan struktur dan negative findings. Karena sebagian genealogy tersebut memakai checkpoint parent yang berbeda dari final direct protocol, hasilnya tidak secara otomatis menjadi final confirmatory result.

### 3.15.3 Bukti final tesis

Klaim utama tesis harus bertumpu pada selected/frozen AF2 configuration yang dibandingkan dengan native YOLO26 menggunakan paired confirmatory protocol. Analisis akhir mencakup:

\[
\boxed{
\text{overall performance}
+
\text{lower-tail behavior}
+
\text{mechanism diagnostics}
+
\text{visualization/error analysis}
+
\text{efficiency}
}
\]

Dengan rancangan ini, istilah **Optimasi** pada judul merujuk pada factorized structural analysis dan limited AF2 parameter sensitivity, sedangkan istilah **Analisis** merujuk pada evaluasi kuantitatif, lower-tail, mekanisme, visualisasi, kesalahan, dan efisiensi.

---

## Guardrails Bab III

Sebelum dokumen final digenerate ke template USU:

1. semua persamaan yang berasal dari literature harus diberi sumber sesuai pedoman;
2. rumus/metric study-defined harus diberi label sebagai formulasi penelitian;
3. exact page/equation citations parent AFAB/LFDet dan YOLO26 harus ditambahkan pada citation audit;
4. nilai parameter sensitivity tambahan tidak boleh ditentukan setelah melihat validation outcome;
5. locked test tidak boleh dipakai untuk model selection;
6. nama metode CAM tidak boleh dikunci sebelum kompatibilitas YOLO26 diverifikasi;
7. hasil candidate lama tidak boleh disamakan dengan final direct-from-pretrained evidence;
8. seluruh tabel/gambar pada DOCX final mengikuti penomoran dan caption template/pedoman USU.