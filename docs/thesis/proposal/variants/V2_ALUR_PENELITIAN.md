# V2 — Diagram Alur Penelitian Multi-Dataset Publik

Status: **WORKING VARIANT — NOT FORMAL**

Diagram ini digunakan khusus untuk rancangan V2.

![Alur penelitian multi-dataset publik](assets/alur_penelitian_multi_dataset_publik.svg){width=12.5cm}

**Gambar V2.1 Alur Penelitian Multi-Dataset Publik**

Urutan penelitian:

1. persiapan **robusta_SNI_Dataset** sebagai dataset utama;
2. pembagian data pelatihan, validasi, dan pengujian secara terkelompok;
3. pembentukan baseline YOLO26n;
4. pengujian variasi prapemrosesan frekuensi-angular pada dataset utama;
5. pemilihan dan pembekuan konfigurasi final $C^*$;
6. persiapan **Capstone, Lulus, dan Niacubilla** sebagai dataset konfirmasi;
7. pelatihan baseline dan $C^*$ secara terpisah pada setiap dataset konfirmasi;
8. evaluasi akhir pada test set setelah protokol dibekukan;
9. analisis hasil pada setiap dataset dan konsistensi lintas dataset;
10. penarikan kesimpulan.

Dataset publik tidak digunakan untuk memilih ulang $C^*$ dan tidak digabung dengan robusta_SNI_Dataset.