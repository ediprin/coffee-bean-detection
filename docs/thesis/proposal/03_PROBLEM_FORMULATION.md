# 1.2–1.6 Problem Formulation — Working Draft

> Status: working draft derived from the frozen foundation, coffee evidence matrix, method bridge matrix, and claim ledger.

## 1.2 Identifikasi Masalah

Berdasarkan kajian literatur dan studi pendahuluan, masalah penelitian diidentifikasi sebagai berikut.

1. **Taxonomy cacat biji kopi yang lebih rinci meningkatkan kesulitan diskriminasi visual.** Penelitian pada coffee-bean classification dan detection menunjukkan bahwa performa yang sangat tinggi pada jumlah kelas terbatas tidak selalu bertahan ketika kategori cacat diperluas menjadi taxonomy yang lebih granular. Beberapa penelitian 15–20 kelas juga menunjukkan perbedaan performa antarkelas yang besar.

2. **Sebagian kategori cacat memiliki karakteristik visual yang saling berdekatan.** Variasi seperti black dan partially black, beberapa kelompok sour/black, floater yang menyerupai normal bean, fungus damage yang menyerupai warna permukaan, serta slight insect damage dengan tanda kecil telah dilaporkan sebagai kategori yang lebih sulit dibedakan.

3. **Metrik agregat dapat menyembunyikan kelas dengan performa rendah.** Nilai mAP atau accuracy keseluruhan tidak selalu merepresentasikan kinerja kategori paling sulit. Karena itu evaluasi perlu memasukkan analisis per-class dan lower-tail performance selain nilai agregat.

4. **Sebagian besar solusi coffee-defect terkini meningkatkan representasi melalui komponen internal model.** Pendekatan yang digunakan mencakup perubahan convolution operator, attention, multiscale feature fusion, Transformer backbone, dan metric/similarity learning. Dalam corpus kopi yang diaudit, input-space frequency-angular preprocessing belum menjadi pendekatan dominan.

5. **Literatur di luar domain kopi menunjukkan bahwa preprocessing dan frequency-domain processing dapat diarahkan untuk meningkatkan downstream detection, tetapi efektivitas transfer ke fine-grained coffee-defect detection belum dapat diasumsikan.** Dengan demikian, frequency-angular preprocessing perlu diuji sebagai hipotesis metodologis melalui perbandingan terkontrol.

6. **Peningkatan detection performance perlu dianalisis lebih lanjut untuk membedakan aspek diskriminasi kelas dan aksesibilitas/lokalisasi proposal.** Literatur object detection menunjukkan bahwa classification confidence dan localization quality tidak identik. Oleh karena itu, peningkatan mAP tidak otomatis dapat disebut sebagai peningkatan kemampuan lokalisasi.

## 1.3 Rumusan Masalah

Penelitian ini dirumuskan melalui tiga pertanyaan utama:

### RQ1

**Apakah preprocessing citra berbasis frekuensi-angular dapat meningkatkan kinerja YOLO26 dalam mendeteksi cacat biji kopi secara fine-grained dibandingkan YOLO26 tanpa preprocessing tersebut?**

Operasionalisasi:

- Macro mAP50–95 sebagai indikator agregat utama;
- mAP50 sebagai indikator tambahan;
- perbandingan dilakukan menggunakan matched training protocol.

### RQ2

**Bagaimana pengaruh preprocessing frekuensi-angular terhadap kelas-kelas cacat yang memiliki kinerja rendah atau sulit dibedakan?**

Operasionalisasi:

- Bottom-3 class mAP50–95;
- Worst-class mAP50–95;
- per-class AP;
- confusion / class error analysis sejauh tersedia.

### RQ3

**Apakah pola perubahan kinerja yang dihasilkan lebih konsisten dengan peningkatan diskriminasi kelas daripada peningkatan aksesibilitas proposal/lokalisasi mentah?**

Operasionalisasi:

- raw proposal accessibility sebagai proxy ketersediaan proposal/lokalisasi mentah;
- localization-conditioned Top-1 classification sebagai indikator diskriminasi setelah syarat lokalisasi terpenuhi;
- correct-decision recall;
- metrik lokalisasi tambahan hanya jika protokol final menyediakan ukuran yang valid.

Batas interpretasi RQ3: diagnostic tersebut membantu atribusi pola, tetapi tidak membuktikan secara kausal bahwa seluruh perubahan berasal dari classification branch atau bahwa kualitas box regression tidak berubah.

## 1.4 Tujuan Penelitian

Tujuan penelitian dirancang berpasangan langsung dengan rumusan masalah.

1. **Mengevaluasi efektivitas preprocessing frekuensi-angular pada YOLO26** untuk fine-grained coffee-defect detection melalui perbandingan dengan native YOLO26 menggunakan kondisi pelatihan yang setara.

2. **Menganalisis dampak preprocessing terhadap difficult dan lower-tail classes** menggunakan Bottom-3, Worst-class, serta per-class performance.

3. **Menganalisis karakter perubahan detection performance** dengan membandingkan indikator aksesibilitas proposal/lokalisasi mentah dan indikator diskriminasi klasifikasi, tanpa mengubah diagnostic tersebut menjadi klaim kausal yang tidak didukung.

4. **Mengevaluasi trade-off akurasi dan efisiensi** melalui parameter count, latency, throughput, dan penggunaan memori, dengan prinsip bahwa parameter-free tidak sama dengan compute-free.

## 1.5 Batasan Penelitian

Untuk menjaga ruang lingkup tesis tetap realistis dan terkontrol, penelitian dibatasi sebagai berikut.

1. Fokus utama adalah **object detection cacat pada green coffee beans**, bukan grading rasa, roasting quality, open-set recognition, atau counting sebagai tujuan utama.

2. Detector utama adalah **YOLO26**, dan kontribusi yang diuji berada pada **input-space preprocessing**, bukan modifikasi backbone, neck, atau detection head.

3. AF2 diposisikan sebagai **parameter-free frequency-angular preprocessing**. Tidak ada klaim bahwa metode ini bebas komputasi.

4. Perbandingan utama menggunakan **native YOLO26 vs AF2-YOLO26** dengan pretrained checkpoint, data split, seed pairing, augmentation, training budget, dan pengaturan lain yang dipasangkan sejauh memungkinkan.

5. Data test yang dikunci tidak digunakan untuk pemilihan metode atau tuning, sesuai protokol repository.

6. Hasil seed tunggal tidak digunakan sebagai dasar klaim final. Studi pendahuluan seed 42 hanya menunjukkan feasibility awal.

7. Klasifikasi dan lokalisasi tidak diperlakukan sebagai satu fenomena tunggal. Interpretasi mekanisme dibatasi pada metrik diagnostik yang benar-benar tersedia.

8. Penelitian tidak mengklaim bahwa frekuensi merupakan bottleneck yang telah terbukti pada coffee defect detection. Frequency-angular processing adalah candidate mechanism yang diuji secara empiris.

9. Klaim novelty dibatasi pada hasil audit literatur. Istilah seperti "pertama" atau "belum pernah" tidak digunakan tanpa verifikasi sistematis tambahan.

## 1.6 Kontribusi yang Diharapkan

### Kontribusi metodologis

Penelitian diharapkan menghasilkan evaluasi terkontrol terhadap penggunaan preprocessing citra berbasis frekuensi-angular sebagai front-end sebelum YOLO26 untuk fine-grained coffee-defect detection.

### Kontribusi analitis

Penelitian tidak hanya mengevaluasi nilai agregat, tetapi juga menganalisis difficult classes melalui Bottom-3, Worst-class, dan per-class AP, sehingga kontribusi metode terhadap lower-tail performance dapat diamati secara lebih jelas.

### Kontribusi diagnostik

Penelitian membandingkan indikator diskriminasi kelas dengan indikator aksesibilitas proposal/lokalisasi mentah sejauh didukung oleh metrik. Dengan demikian, peningkatan detection score tidak langsung diasumsikan berasal dari lokalisasi yang lebih baik maupun secara kausal hanya dari klasifikasi.

### Kontribusi rekayasa

AF2 tidak menambah learned preprocessing parameters pada detector. Penelitian akan tetap mengukur latency, throughput, dan memory overhead agar trade-off implementasi dinilai secara lengkap.

---

## Research logic in one line

```text
coffee fine-grained difficulty
→ need stronger discriminative input/representation
→ frequency-angular preprocessing is a plausible but unvalidated candidate
→ controlled YOLO26 comparison
→ aggregate + tail + classification/proposal-accessibility diagnosis
```

## Guardrail

The proposal should say **"menguji"**, **"menganalisis"**, and **"mengevaluasi"** before the final experiments are complete. Avoid replacing those verbs with **"membuktikan"** or **"mengatasi"** at proposal stage.
