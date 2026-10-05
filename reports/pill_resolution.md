# `pill` için çözünürlük, model v2 ve geri alma provası — 2026-10-05

## Soru

[Eşik raporunda](threshold_calibration.md) `pill`, kusurların yalnızca yaklaşık yarısını
yakalıyordu. Kaçanlar çoğunlukla ince çatlaklar (`crack`) ve bozuk baskılardı
(`faulty_imprint`). Bu ayrıntılar 256 × 256'ya küçültülürken kayboluyor olabilir. Daha
yüksek çözünürlük `pill`'i iyileştirir mi?

## Protokol

Sonuçlara bakmadan önce [`scripts/compare_resolutions.py`](../scripts/compare_resolutions.py)
içine yazıldı:

- Adaylar: canlıdaki 256 bankası ve aynı ayarlarla (coreset 16.384, `neighborhood=3`,
  `border=2`, seed 42) 320 ve 384'te kurulan iki yeni banka.
- `pill` test görüntüleri, sağlam ve kusurlu ayrı ayrı olmak üzere ikiye bölündü (seed 42).
  Adaylar yalnızca **doğrulama** yarısında, image AUROC ile yarıştı. Eşitlikte küçük
  çözünürlük kazanır (servis etmesi daha ucuz).
- Kazanan ve canlıdaki 256 bankası **final** yarısında bir kez ölçüldü.

`run_pipeline` her koşuda AUROC'u bütün test kümesinde hesaplayıp MLflow'a yazdığı için
bu sayılar seçimden önce görüldü. Seçim kuralı önceden yazılı olduğundan ve yalnızca
doğrulama yarısına baktığından bu, kararı etkilemedi.

## Sonuç

| Çözünürlük | Doğrulama image AUROC | Final yarısı image AUROC |
| --- | ---: | ---: |
| 256 (v1) | 0,933 | 0,953 |
| **320** | **0,963** | **0,963** |
| 384 | 0,950 | |

**320 kazandı.** Final yarısındaki iyileşme (+0,010) doğrulamadakinden (+0,030) küçük.
Kazanan doğrulamada seçildiği için orada biraz iyimser çıkması beklenir; final yarısı daha
gerçekçi tahmindir. 384'ün 320'den kötü çıkması, çözünürlük arttıkça hapın yüzeyindeki
doğal beneklerin de "yabancı" görünmeye başlamasıyla açıklanabilir; bu ayrıca incelenmedi.

Eşik yeni bankayla, aynı kuralla (ayrılan sağlam eğitim görüntülerinin en yüksek puanı)
yeniden hesaplandı: 17,55 → 17,85. Bütün test kümesinde (bilgi amaçlı, seçimde
kullanılmadı) kusur tiplerine göre yakalanan:

| Kusur tipi | v1 (256) | v2 (320) |
| --- | ---: | ---: |
| `crack` | 5 / 26 | 9 / 26 |
| `faulty_imprint` | 4 / 19 | 9 / 19 |
| `color` | 11 / 25 | 17 / 25 |
| `scratch` | 14 / 24 | 17 / 24 |
| `contamination` | 13 / 21 | 15 / 21 |
| `combined` | 13 / 17 | 15 / 17 |
| `pill_type` | 9 / 9 | 9 / 9 |
| sağlam (yanlış alarm) | 0 / 26 | 0 / 26 |

Recall 0,489'dan 0,645'e çıktı, yanlış alarm 0'da kaldı. En büyük kazanç hedeflenen ince
kusurlarda (`faulty_imprint` ve `crack`).

## Model v2

- Bankalar artık hangi çözünürlükte kurulduklarını saklıyor; eski bankalar 256 sayılıyor.
  Servis her kategoriyi kendi bankasının çözünürlüğünde işliyor.
- v2 = v1'in 14 bankası + 320'de kurulan `pill` bankası. MLflow'da `patchcore-mvtec` v2
  olarak kayıtlı; kaynak run `pill-320`, etiketler `base_version=1` ve
  `change=pill image_size 256 to 320`. Drift referansı v2 için yeniden üretildi.

## Canlıya alma ve geri alma provası

Her adım: MLflow'da `production` etiketini taşı, `python -m scripts.deploy_model` çalıştır.

| Adım | Süre | Sonuç |
| --- | ---: | --- |
| v1 → v2 | 97 sn | v2 klasörü yüklendi (17 dosya); `/model` "2"; drift görevi v2 referansında |
| v2 → v1 (geri alma) | 49 sn | Yükleme atlandı; `/model` "1"; drift görevi v1 referansında |
| v1 → v2 (son) | 70 sn | `/model` "2" |

Süreler, etiketi taşımaktan Cloud Run'ın yeni revizyonu hazır bildirmesine kadar ölçüldü.
Geri alma daha hızlı, çünkü sürümün dosyaları bucket'ta zaten duruyor ve değişmiyor.

Aynı çatlaklı hap (`data/data_15/002-28.png`) canlı serviste v2'de kusurlu (puan 18,37,
eşik 17,85), geri alınan v1'de sağlam (puan 16,87, eşik 17,55) çıktı. Prova sonunda,
protokolün kazananı olan **v2 canlıda** bırakıldı.

## Sınırlar

- Tek kategori, tek seed ve tek bölme. Final yarısında 0,01'lik fark küçük; başka bir
  bölmede daralabilir veya genişleyebilir.
- Diğer 14 kategori 256'da kaldı; çözünürlük onlar için denenmedi.
- 320'de `pill` için hem bellek hem çıkarım süresi 256'dan biraz fazla.
