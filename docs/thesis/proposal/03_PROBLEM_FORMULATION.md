# 1.2–1.6 Problem Formulation — Working Draft

> Status: aligned with the Hong-adapted Bab III optimization design.

## 1.2 Identifikasi Masalah

Berdasarkan kajian literatur dan studi pendahuluan, masalah penelitian diidentifikasi sebagai berikut.

1. Taxonomy cacat biji kopi yang lebih rinci meningkatkan kesulitan diskriminasi visual dan menunjukkan perbedaan performa antarkelas.
2. Sebagian kategori cacat memiliki karakteristik visual yang saling berdekatan sehingga kelas tertentu lebih sulit dibedakan.
3. Metrik agregat dapat menyembunyikan kelas dengan performa rendah sehingga analisis per-class dan lower-tail diperlukan.
4. Sebagian besar solusi coffee-defect yang diaudit meningkatkan representasi melalui komponen internal model, sedangkan input-space frequency-angular preprocessing belum menjadi pendekatan dominan pada corpus tersebut.
5. Literatur di luar domain kopi menunjukkan bahwa preprocessing dan frequency-domain processing dapat memengaruhi downstream detection, tetapi efektivitas transfer ke fine-grained coffee-defect detection tetap harus diuji.
6. AF2 reference terdiri atas beberapa keputusan desain—windowing, representasi angular, struktur radial, threshold, dan pemrosesan channel—yang kontribusinya perlu dianalisis terpisah sebelum konfigurasi final dipilih.
7. Peningkatan detection performance perlu dianalisis dengan membedakan indikator diskriminasi kelas dari indikator aksesibilitas proposal/lokalisasi mentah.
8. AF2 tidak memiliki learned preprocessing parameters, tetapi tetap menambah komputasi sehingga accuracy-efficiency trade-off perlu diukur.

## 1.3 Rumusan Masalah

Penelitian ini dirumuskan melalui empat pertanyaan utama.

### RQ1

**Bagaimana pengaruh keputusan desain utama AF2 terhadap kinerja fine-grained coffee-defect detection, dan konfigurasi preprocessing frekuensi-angular seperti apa yang paling layak dipilih melalui analisis terfaktor dan sensitivity analysis?**

Operasionalisasi:

1. membandingkan `AF2C`, `AF2WIN`, `AF2ORI`, `AF2POL`, `AF2SOFT`, dan `AF2LUM` secara satu-faktor-pada-satu-waktu;
2. menggunakan Macro mAP50–95 sebagai primary selection metric;
3. menggunakan Bottom-3 dan Worst-class sebagai tail/safety indicators;
4. melaporkan latency sebagai engineering trade-off;
5. membekukan konfigurasi sebelum eksperimen konfirmatori dan tidak menggunakan locked test untuk selection.

### RQ2

**Apakah konfigurasi AF2 yang telah dipilih dapat meningkatkan kinerja YOLO26 dalam mendeteksi cacat biji kopi secara fine-grained dibandingkan YOLO26 tanpa preprocessing pada eksperimen konfirmatori yang dipasangkan?**

Operasionalisasi:

1. native YOLO26n versus selected/frozen AF2-YOLO26n;
2. official pretrained source dan target-head initialization yang dipadankan;
3. Macro mAP50–95 sebagai indikator agregat utama;
4. mAP50 dan mAP50–95 sebagai indikator tambahan;
5. repeated paired seeds.

### RQ3

**Bagaimana pengaruh preprocessing frekuensi-angular terhadap kelas-kelas cacat yang memiliki kinerja rendah atau sulit dibedakan?**

Operasionalisasi:

1. Bottom-3 class mAP50–95;
2. Worst-class mAP50–95;
3. per-class AP;
4. confusion/error analysis;
5. paired rescue-regression analysis.

### RQ4

**Apakah pola perubahan kinerja yang dihasilkan lebih konsisten dengan peningkatan diskriminasi kelas daripada peningkatan aksesibilitas proposal/lokalisasi mentah?**

Operasionalisasi:

1. raw proposal accessibility;
2. localization-conditioned Top-1 classification;
3. correct-decision recall;
4. visualisasi input/spektral dan activation visualization bila kompatibilitas dengan YOLO26 telah diverifikasi.

Batas interpretasi RQ4: diagnostic membantu atribusi pola, tetapi tidak membuktikan secara kausal bahwa seluruh perubahan berasal dari classification branch atau bahwa kualitas box regression tidak berubah.

## 1.4 Tujuan Penelitian

1. **Menganalisis dan mengoptimasi keputusan desain AF2** melalui factorized structural analysis dan limited parameter sensitivity sehingga konfigurasi final dipilih berdasarkan development protocol yang dibekukan.
2. **Mengevaluasi efektivitas konfigurasi AF2 terpilih pada YOLO26** melalui eksperimen konfirmatori dengan native YOLO26 menggunakan kondisi initialization dan pelatihan yang dipasangkan.
3. **Menganalisis dampak preprocessing terhadap difficult dan lower-tail classes** menggunakan Bottom-3, Worst-class, per-class AP, dan paired error analysis.
4. **Menganalisis karakter perubahan detection performance** melalui proposal-accessibility diagnostic, localization-conditioned classification, visualisasi, dan error analysis tanpa memperluasnya menjadi klaim kausal yang tidak didukung.
5. **Mengevaluasi trade-off akurasi dan efisiensi** melalui parameter count, latency, throughput, dan penggunaan memori.

## 1.5 Batasan Penelitian

1. Fokus utama adalah object detection cacat pada green coffee beans, bukan grading rasa, roasting quality, open-set recognition, atau counting sebagai tujuan utama.
2. Detector utama adalah YOLO26; kontribusi yang diuji berada pada input-space preprocessing, bukan modifikasi backbone, neck, atau detection head.
3. AF2 diposisikan sebagai parameter-free frequency-angular preprocessing, bukan sebagai metode bebas komputasi.
4. Optimasi utama dibatasi pada keputusan desain AF2 yang dapat diisolasi secara satu-faktor-pada-satu-waktu dan limited parameter sensitivity; penelitian tidak melakukan broad module stacking.
5. Kandidat struktural utama adalah `AF2C`, `AF2WIN`, `AF2ORI`, `AF2POL`, `AF2SOFT`, dan `AF2LUM`. `PCG1` dan `WAV1` hanya optional mechanistic comparators.
6. Perbandingan konfirmatori menggunakan native YOLO26 versus selected AF2-YOLO26 dengan pretrained checkpoint, data split, seed pairing, augmentation, training budget, dan pengaturan lain yang dipasangkan.
7. Locked test tidak digunakan untuk model selection atau tuning.
8. Hasil seed tunggal tidak digunakan sebagai dasar klaim final; seed 42 hanya feasibility evidence.
9. Genealogy factorization lama yang memakai parent checkpoint berbeda diperlakukan sebagai development evidence, bukan final direct-from-pretrained confirmation.
10. Interpretasi mekanisme dibatasi pada diagnostic yang benar-benar tersedia.
11. Penelitian tidak mengklaim frekuensi sebagai bottleneck coffee defect detection yang telah terbukti.
12. Klaim novelty dibatasi pada hasil audit literatur dan tidak menggunakan klaim global “pertama” tanpa verifikasi sistematis.

## 1.6 Kontribusi yang Diharapkan

### Kontribusi metodologis

Evaluasi terkontrol terhadap preprocessing frekuensi-angular sebagai front-end YOLO26 serta factorized analysis untuk memberi dasar eksplisit terhadap pemilihan konfigurasi AF2.

### Kontribusi analitis

Analisis aggregate, Bottom-3, Worst-class, per-class AP, dan paired error transitions untuk menunjukkan kelas yang terbantu maupun mengalami regresi.

### Kontribusi diagnostik

Perbandingan indikator diskriminasi kelas dengan proposal/localization accessibility serta dukungan visualisasi input/spektral dan error analysis, dengan batas interpretasi non-kausal.

### Kontribusi rekayasa

Pengukuran parameter count, latency, throughput, dan memory overhead sehingga parameter-free tidak disamakan dengan compute-free.

---

## Research logic in one line

```text
coffee fine-grained difficulty
→ frequency-angular preprocessing as candidate mechanism
→ factorized AF2 optimization
→ method freeze
→ paired native-vs-AF2 confirmation
→ aggregate + tail + mechanism + visualization/error + efficiency analysis
```

## Guardrail

Pada tahap proposal, gunakan kata *menguji*, *menganalisis*, *mengoptimasi berdasarkan development protocol*, dan *mengevaluasi*. Hindari klaim *optimal secara global*, *mengatasi*, atau *membuktikan* sebelum evidence final tersedia.