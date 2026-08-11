# SNI-21 Structured-Target Support Audit Protocol

## Tujuan

Audit ini memeriksa apakah target terstruktur SNI-21 yang telah diformalkan
memiliki dukungan anotasi development yang memadai sebelum arsitektur atau loss
baru dirancang. Audit tidak melatih model, tidak menjalankan inferensi, dan
tidak membuka test.

## Data yang diizinkan

- Faruq-v3 grouped development `train` dan `val`.
- `faruq_grouped_manifest.json` untuk menghitung identitas sumber independen.
- Ontologi terkunci `configs/sni21/structured_ontology_v1.yaml`.

Jika folder `test` tersedia, audit wajib berhenti.

## Unit dukungan

Setiap nilai target dilaporkan sebagai:

- jumlah instance;
- jumlah gambar berbeda;
- jumlah `group_id` sumber independen;
- dukungan terpisah pada train dan validation.

Ambang audit v1 adalah minimal 50 instance dan 25 group pada train, serta 10
instance dan 10 group pada validation. Ambang ini adalah gate kelayakan awal,
bukan bukti bahwa target dapat dikenali dari RGB.

## Gate semantik

- `physical_size_mm` tetap diblokir sampai ada referensi skala terkalibrasi.
- `relative_completeness` memerlukan review ahli domain karena bergantung pada
  pembanding bentuk utuh.
- `positive_flag` merupakan pengawasan positif parsial. Atribut yang tidak
  disebut pada kelas bukan label negatif.
- Nilai RGB-sensitive baru boleh masuk protokol model setelah dukungan
  statistik dan observabilitasnya disetujui.

## Batas keputusan

Output selalu `AUDIT_COMPLETE_NO_TRAINING_AUTHORIZATION`. Hasil audit hanya
menentukan target mana yang dapat dipertimbangkan pada protokol berikutnya;
hasil ini tidak mengizinkan training, multi-seed, atau akses test.
