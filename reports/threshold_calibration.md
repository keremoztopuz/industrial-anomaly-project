# `is_anomaly` eşiği — 2026-10-04

## Sorun

Servis şimdiye kadar yalnızca bir anomali puanı döndürüyordu. Bir kullanıcı için
"puan 17,9" tek başına bir şey söylemez; "bu parça kusurlu mu?" sorusunun cevabı
için her kategoriye bir eşik gerekir: puan eşiğin üstündeyse kusurlu, değilse sağlam.

Eşik, puanların ölçeği kategoriden kategoriye değiştiği için tek bir sayı olamaz.
Örneğin sağlam `hazelnut` görüntülerinin puanı 17 civarındayken sağlam `zipper`
görüntülerinin puanı 10 civarındadır.

## Yöntem

MVTec AD'de doğrulama kümesi yoktur ve eşik test kümesine bakılarak seçilirse
sonuç iyimser olur. Bu yüzden eşik yalnızca eğitim kümesinin sağlam görüntüleriyle
seçildi:

1. Her kategorinin sağlam eğitim görüntüleri seed 42 ile karıştırıldı ve %20'si
   ayrıldı (en az 1 görüntü, yukarı yuvarlanarak).
2. Kalan %80 ile, canlıdaki bankayla aynı ayarlarda yeni bir banka kuruldu
   (coreset, 16.384 patch, `neighborhood=3`, `border=2`, izdüşüm seed 42).
3. Ayrılan görüntüler bu bankayla puanlandı. **Eşik, bu puanların en yükseğidir.**
   Yani ayrılan sağlam görüntülerin hiçbiri kusurlu sayılmaz.
4. Eşik, canlıdaki (%100 ile kurulmuş) bankaya uygulanır. Test kümesi yalnızca bu
   noktadan sonra, sonucu raporlamak için kullanıldı; eşik ona göre değiştirilmedi.

Karar kuralı: `anomaly_score > threshold` ise `is_anomaly = true`.

%80 ve %100 bankalarının puanları neredeyse aynıdır. `toothbrush`'ta iki banka
aynı 42 test görüntüsünün çoğuna 0,5'ten az farkla puan verdi (ör. 19,3 ve 19,2);
en büyük fark 1,0 oldu (20,7 ve 19,7). Bu yüzden
%80 bankasından bulunan eşiği %100 bankasına uygulamak makul bir yaklaşımdır.

## Sonuçlar

Recall: kusurlu test görüntülerinin yakalanan oranı. Yanlış alarm: sağlam test
görüntülerinin kusurlu sayılan oranı. Image AUROC, [kenar raporundaki](border_exclusion.md)
resmi koşudandır ve eşikten bağımsızdır.

| Kategori | Ayrılan sağlam | Eşik | Recall | Yanlış alarm | Kaçan / kusurlu | Yanlış alarm / sağlam | Image AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bottle | 42 | 12,82 | 1,000 | 0,050 | 0 / 63 | 1 / 20 | 1,0000 |
| cable | 45 | 17,94 | 0,935 | 0,052 | 6 / 92 | 3 / 58 | 0,9921 |
| capsule | 44 | 13,60 | 0,670 | 0,000 | 36 / 109 | 0 / 23 | 0,9733 |
| carpet | 56 | 12,72 | 0,978 | 0,429 | 2 / 89 | 12 / 28 | 0,9819 |
| grid | 53 | 15,42 | 0,860 | 0,048 | 8 / 57 | 1 / 21 | 0,9758 |
| hazelnut | 79 | 19,80 | 1,000 | 0,000 | 0 / 70 | 0 / 40 | 1,0000 |
| leather | 49 | 13,41 | 1,000 | 0,000 | 0 / 92 | 0 / 32 | 1,0000 |
| metal_nut | 44 | 17,47 | 0,968 | 0,000 | 3 / 93 | 0 / 22 | 0,9985 |
| pill | 54 | 17,55 | 0,489 | 0,000 | 72 / 141 | 0 / 26 | 0,9444 |
| screw | 64 | 14,60 | 0,630 | 0,073 | 44 / 119 | 3 / 41 | 0,9250 |
| tile | 46 | 18,87 | 0,881 | 0,000 | 10 / 84 | 0 / 33 | 0,9996 |
| toothbrush | 12 | 12,47 | 1,000 | 0,417 | 0 / 30 | 5 / 12 | 0,9194 |
| transistor | 43 | 16,96 | 0,975 | 0,033 | 1 / 40 | 2 / 60 | 0,9954 |
| wood | 50 | 17,72 | 0,950 | 0,105 | 3 / 60 | 2 / 19 | 0,9868 |
| zipper | 48 | 13,69 | 0,924 | 0,094 | 9 / 119 | 3 / 32 | 0,9632 |
| **Toplam** | | | **0,846** | **0,069** | **194 / 1.258** | **32 / 467** | |

Toplam satırı bütün test görüntülerini birlikte sayar (makro ortalama değildir).
Bütün test görüntülerinde doğru karar oranı 0,869'dur.

## Yorum

- **Altı kategoride sonuç iyi:** `bottle`, `hazelnut`, `leather`, `metal_nut`,
  `transistor` ve `cable`'da kusurların %93'ünden fazlası yakalanıyor, yanlış alarm
  %5 civarında veya altında.
- **Kaçırılan kusurlar (`pill` 0,489, `screw` 0,630, `capsule` 0,670):** Bu
  kategorilerde yanlış alarm neredeyse yok, ama küçük kusurların çoğu eşiğin altında
  kalıyor. AUROC yüksek olduğu için (`pill` 0,94, `capsule` 0,97) puanların sıralaması
  iyi; sorun eşiğin yerinde. Eşik "ayrılan en yüksek sağlam puan" olduğu için
  temkinlidir ve küçük kusurları sağlam sayar.
- **Yanlış alarmlar (`carpet` 0,429, `toothbrush` 0,417):** Test kümesindeki bazı
  sağlam görüntüler, ayrılan sağlam eğitim görüntülerinin hepsinden yüksek puan aldı.
  `toothbrush`'ta yalnızca 12 görüntü ayrılabildi; bu kadar az görüntüyle puan
  dağılımının üst ucu iyi kestirilemez. Test kümesindeki iki sağlam görüntü 18,5 ve
  19,2 puan aldı; bunlar kusurlu görüntülerin çoğuyla aynı aralıkta. `carpet`'te
  56 görüntü ayrılmasına rağmen yanlış alarm yüksek; bu, test kümesindeki sağlam
  halıların eğitim kümesindekilerden biraz farklı olduğuna işaret ediyor olabilir.
  Bu ihtimal ayrıca incelenmedi.

Kısacası tek bir kural her kategoride aynı dengeyi kurmuyor: bazı kategorilerde
temkinli, bazılarında fazla hassas. Kaçan kusurla yanlış alarm arasındaki tercih
bir iş kararıdır (bir kusuru kaçırmak mı daha pahalı, sağlam parçayı atmak mı?).
Bu rapor bu tercihi yapmaz; yalnızca, test kümesine bakmadan seçilmiş tek bir
kuralın sonucunu gösterir.

## Sınırlar

- Ayrılan küme küçüktür (12–79 görüntü). En yüksek değer, puan dağılımının üst
  ucunu olduğundan düşük gösterir; daha büyük bir kalibrasyon kümesi veya
  k-katlı bölme daha kararlı bir eşik verirdi. 15 kategoride tam bir koşu M4'te
  yaklaşık 55 dakika sürdüğü için k-katlı bölme (yaklaşık 5 kat süre) denenmedi.
- Sonuçlar tek seed'e ve tek bir bölmeye dayanır.
- `border=2` test sonuçlarına bakılarak seçildiği için ([kenar raporu](border_exclusion.md)),
  test puanları bir miktar iyimser olabilir. Eşiğin kendisi test kümesine bakılmadan seçildi.

## Servis

[`scripts/calibrate_thresholds.py`](../scripts/calibrate_thresholds.py), bankaların
yanına iki dosya yazar:

- `thresholds.json`: kategori başına eşik ve kalibrasyon bilgisi. Servis bu dosyayı
  açılışta `MODEL_DIR`'dan okur ve `/predict` cevabına `threshold` ve `is_anomaly`
  ekler. Dosya yoksa ya da kategori dosyada yoksa iki alan da `null` olur.
- `threshold_evaluation.json`: yukarıdaki tablonun ham sayıları.

Canlı servis için `thresholds.json`, bankalarla aynı Cloud Storage bucket'ına kondu.

Kalibrasyon `bf1da8f` commit'i ve `scripts/calibrate_thresholds.py` ile 2026-10-04'te,
Apple MPS üzerinde yaklaşık 55 dakikada çalıştırıldı. `thresholds.json` SHA256:
`bd872203f8ccf8ede2ed641f7d6712558b894017afcbdada4eda7413c1f6c4ac`. Yeniden üretmek için:

```sh
.venv/bin/python -m scripts.calibrate_thresholds \
  --dataset-root data/mvtec-ad \
  --model-dir artifacts/border-exclusion/coreset-16384-n3-b2/patchcore \
  --holdout-fraction 0.2 --seed 42 --device mps
```
