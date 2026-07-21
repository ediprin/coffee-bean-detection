# SNI 300 g Copy-Paste Preview

Preset `sni_spread` membuat scene sintetis yang meniru tampilan hamparan sampel
kopi berjumlah tinggi: objek tersebar di seluruh bidang dan mayoritas satu lapis.
Preset ini terpisah dari VA-DCP padat lama dan tidak mengubah perilaku default.

Istilah “300 g” adalah target skenario, **bukan konversi otomatis massa ke 220–340
biji**. Rentang awal mengikuti kepadatan visual gambar referensi dan harus diganti
dengan distribusi jumlah biji yang dihitung dari foto 300 g nyata.

## Desain

- 220–340 biji per scene secara default.
- Kanvas potret 3:4 dengan long side 768 px secara default.
- Ukuran objek 2,5–5,5% dari long side canvas.
- Mode scene: 100% `spread` di seluruh bidang, tanpa gumpalan `cluster`/`pile`.
- A2 menargetkan 70% clear, 25% mild, 5% severe, dan 0% extreme.
- Distribusi kelas mengikuti frekuensi label pada train nyata, bukan dipaksa sama.
- Label polygon YOLO-segmentation dipakai langsung sebagai alpha mask. Label box
  biasa tetap menggunakan estimasi foreground.
- Val/test nyata tidak diubah. Runner preview bahkan tidak menyalinnya agar cepat.

## Preview Colab

Gunakan `notebooks/SNI_300g_CopyPaste_Preview_Colab.ipynb`. Notebook membuat
empat A1 dan empat A2, mencetak progres `[1/4]` sampai `[4/4]`, menghasilkan raw
contact sheet dan overlay anotasi, serta tidak menjalankan training.

CLI ekuivalen:

```bash
python -u -m coffee_detector.run_sni_spread_preview \
  --real-data-root /content/coffee-defect-roboflow \
  --object-library /content/coffee-object-library \
  --output-root /content/drive/MyDrive/coffee-bean-detection/sni-300g-preview/v1 \
  --images 4 \
  --objects-min 220 \
  --objects-max 300 \
  --canvas-size 768 \
  --seed 42
```

## Batas klaim

Copy-paste ini adalah augmentasi terkontrol, bukan pengganti data nyata. Keputusan
akhir tetap harus memakai validation/test foto sampel 300 g nyata yang tidak
berasal dari cutout train. Preview harus ditolak bila skala, tepi cutout, bayangan,
warna, atau kepadatan terlihat tidak realistis.
