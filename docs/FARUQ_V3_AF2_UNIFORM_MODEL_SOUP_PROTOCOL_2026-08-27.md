# Faruq-v3 AF2 Uniform Model Soup Protocol

Tanggal pembekuan: 27 Agustus 2026.

## Pertanyaan

Apakah perataan bobot tiga checkpoint AF2 yang telah dikonfirmasi dapat
menstabilkan kelas terbawah tanpa mengubah arsitektur, menambah parameter, atau
melakukan training tambahan?

## Desain terkunci

- Sumber: checkpoint AF2 seed 42, 123, dan 2026 dari konfirmasi paired yang
  telah selesai.
- Koefisien ditetapkan **uniform 1/3, 1/3, 1/3** sebelum melihat hasil soup.
- Tidak ada greedy soup, pencarian koefisien, fine-tuning, atau pemilihan dengan
  validation.
- Hanya tensor floating-point yang dirata-ratakan (akumulasi float64 lalu
  dikembalikan ke dtype asli). Buffer integer/bool harus identik.
- Class model, config AF2, YAML, names, state schema, shape, dtype, dan jumlah
  parameter wajib identik.
- Evaluasi hanya pada validation Faruq-v3 grouped. Test tidak dipulihkan atau
  dibaca.

## Metrik dan gate

Referensi adalah rerata aritmetika hasil tiga seed AF2 yang telah dibekukan:

- Macro mAP50–95: 0.8793765274
- Bottom-3 class mAP50–95: 0.7937036280
- Worst-class mAP50–95: 0.7815268371

Soup berstatus `RETAIN` hanya jika seluruh syarat berikut terpenuhi:

1. Macro tidak lebih rendah dari rerata tiga seed AF2.
2. Bottom-3 tidak lebih rendah dari rerata tiga seed AF2.
3. Worst-class tidak lebih rendah dari rerata tiga seed AF2.
4. Minimal salah satu metrik tail (Bottom-3/Worst) naik 0,5 poin persentase.
5. Seluruh 21 kelas tersedia pada validation dan test tetap tertutup.

Jika gagal, AF2 asli tetap metode utama dan tidak ada tuning soup lanjutan.
Model soup ini merupakan kontrol stabilisasi/optimisasi bobot, bukan arsitektur
baru dan bukan pengganti kontribusi mekanistik AF2.

Dasar metodologis model soup: Wortsman et al., *Model soups: averaging weights
of multiple fine-tuned models improves accuracy without increasing inference
time* (ICML 2022). Studi ini memakai varian uniform agar tidak terjadi
pemilihan bobot berdasarkan validation.

## Artefak

- Output root: `experiments/faruq-v3-af2-uniform-soup-v1`
- Checkpoint: `AF2_UNIFORM_SOUP/weights/best.pt`
- Static audit: `static_audit.json`
- Evaluation: `val_reports/af2_uniform_soup_evaluation.json`
- Decision: `val_reports/af2_uniform_soup_decision.json`
