# Banka boyutu taraması — 2026-10-01

## Deney

MVTec AD'nin 15 kategorisinde banka kapasitesini ve patch seçimini karşılaştırdım.
Bütün yeni koşularda görüntü boyutu 256 × 256, batch 8, izdüşüm 256 boyut,
seed 42 ve Apple MPS kullanıldı. ImageNet ağırlıklı Wide ResNet-50-2 sabit kaldı;
`layer2` ve `layer3` özellikleri kullanıldı. Rastgele seçim normal patch'lerden
örnek alır. Coreset seçiminde en fazla banka kapasitesinin dört katı adaydan
en uzak patch'ler sırayla seçilir. Model eğitimi veya hiperparametre araması yoktur.

2.048 rastgele sütunu önceki [baseline raporundan](baseline.md) alınmıştır; bu
dalda yeniden çalıştırılmadı. Diğer beş koşu aynı kaynak commit ile 15 kategoride
yeniden çalıştırıldı. AUROC 0–1 aralığındadır; yüksek değer daha iyidir. Makro
ortalama her kategoriye eşit ağırlık verir. Bunlar tek seed sonuçlarıdır;
belirsizlik aralığı veya tekrarlı koşu yoktur. Yeni koşuların makro ortalamaları
tablodaki yuvarlanmış değerlerden değil, `metrics.json` içindeki tam değerlerden
hesaplandı.

## Sonuçlar

### Image AUROC

| Kategori | 2.048 rastgele | 2.048 coreset | 8.192 rastgele | 8.192 coreset | 16.384 rastgele | 16.384 coreset |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bottle | 0.9802 | 0.9889 | 0.9865 | 0.9944 | 0.9897 | 0.9968 |
| cable | 0.7586 | 0.8504 | 0.8203 | 0.8583 | 0.8371 | 0.8896 |
| capsule | 0.7335 | 0.8608 | 0.8488 | 0.9346 | 0.9047 | 0.9254 |
| carpet | 0.9262 | 0.9579 | 0.9486 | 0.9575 | 0.9498 | 0.9623 |
| grid | 0.6850 | 0.6667 | 0.6207 | 0.6449 | 0.5881 | 0.7243 |
| hazelnut | 0.9600 | 0.9746 | 0.9711 | 0.9704 | 0.9689 | 0.9954 |
| leather | 0.9997 | 0.9993 | 0.9993 | 1.0000 | 0.9986 | 1.0000 |
| metal_nut | 0.8915 | 0.9536 | 0.9531 | 0.9790 | 0.9765 | 0.9897 |
| pill | 0.7717 | 0.8909 | 0.8699 | 0.9307 | 0.9135 | 0.9285 |
| screw | 0.5919 | 0.6300 | 0.6460 | 0.8686 | 0.7920 | 0.9377 |
| tile | 0.8669 | 0.9138 | 0.9019 | 0.9080 | 0.8990 | 0.9098 |
| toothbrush | 0.8889 | 0.8611 | 0.8556 | 0.8694 | 0.8556 | 0.8667 |
| transistor | 0.6804 | 0.8075 | 0.7863 | 0.8800 | 0.8642 | 0.9117 |
| wood | 0.9649 | 0.9833 | 0.9737 | 0.9825 | 0.9798 | 0.9939 |
| zipper | 0.8464 | 0.8952 | 0.8850 | 0.9407 | 0.9262 | 0.9443 |
| **Makro ortalama** | **0.8364** | **0.8823** | **0.8711** | **0.9146** | **0.8963** | **0.9317** |

### Pixel AUROC

| Kategori | 2.048 rastgele | 2.048 coreset | 8.192 rastgele | 8.192 coreset | 16.384 rastgele | 16.384 coreset |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bottle | 0.9807 | 0.9835 | 0.9828 | 0.9848 | 0.9833 | 0.9852 |
| cable | 0.9360 | 0.9531 | 0.9481 | 0.9621 | 0.9536 | 0.9667 |
| capsule | 0.9717 | 0.9790 | 0.9773 | 0.9831 | 0.9792 | 0.9840 |
| carpet | 0.9786 | 0.9860 | 0.9825 | 0.9867 | 0.9835 | 0.9870 |
| grid | 0.9197 | 0.9518 | 0.9417 | 0.9719 | 0.9532 | 0.9768 |
| hazelnut | 0.9807 | 0.9838 | 0.9828 | 0.9858 | 0.9842 | 0.9865 |
| leather | 0.9917 | 0.9926 | 0.9923 | 0.9929 | 0.9925 | 0.9929 |
| metal_nut | 0.9391 | 0.9613 | 0.9549 | 0.9679 | 0.9597 | 0.9717 |
| pill | 0.9169 | 0.9408 | 0.9291 | 0.9491 | 0.9338 | 0.9517 |
| screw | 0.9687 | 0.9777 | 0.9792 | 0.9861 | 0.9830 | 0.9886 |
| tile | 0.9107 | 0.9339 | 0.9178 | 0.9374 | 0.9211 | 0.9393 |
| toothbrush | 0.9852 | 0.9867 | 0.9883 | 0.9883 | 0.9885 | 0.9883 |
| transistor | 0.7865 | 0.8352 | 0.8238 | 0.8623 | 0.8403 | 0.8740 |
| wood | 0.9249 | 0.9349 | 0.9304 | 0.9386 | 0.9325 | 0.9395 |
| zipper | 0.9667 | 0.9779 | 0.9727 | 0.9815 | 0.9744 | 0.9820 |
| **Makro ortalama** | **0.9439** | **0.9585** | **0.9536** | **0.9652** | **0.9575** | **0.9676** |

Aynı kapasitede coreset − rastgele makro farkı 2.048 için image +0.0459,
pixel +0.0147; 8.192 için +0.0435 ve +0.0116; 16.384 için +0.0355 ve
+0.0101 oldu. 16.384 coreset, 8.192 coreset'e göre image +0.0171 ve pixel
+0.0024 sağladı; çalışma süresi yaklaşık 4,8 katına çıktı. En yüksek makro
AUROC 16.384 coreset'te. Kategori sonuçları her artışta düzenli yükselmiyor:
örneğin `grid` image AUROC'u rastgele seçimde banka büyüdükçe düştü.

## Süre ve manifest

Süreler `time -p` ile ölçülen gerçek duvar saati süresidir; Prefect'in açılıp
kapanması da dahildir. Önceki 2.048 rastgele koşusunun 194,57 sn süresi
[baseline raporundan](baseline.md) gelir.

| Koşu | Süre (sn) | Manifest zamanı (UTC) |
| --- | ---: | --- |
| 2.048 coreset | 443,22 | 2026-10-01 14:31:46 |
| 8.192 rastgele | 266,40 | 2026-10-01 13:00:02 |
| 8.192 coreset | 771,07 | 2026-10-01 13:22:07 |
| 16.384 rastgele | 249,88 | 2026-10-01 13:04:45 |
| 16.384 coreset | 3.679,86 | 2026-10-01 14:23:43 |

Beş yeni manifestte kaynak commit `c0a0c8d35dd11c94c8d6f9ee00669cc8bc323b35`,
veri `samples.json` SHA256 değeri
`4382cbf3b69e515b808d2e12232a84df042ed067243ba304d5f6a820e22375e0`.
`git_dirty=true`: koşular sırasında takip edilen `.nodeterm/project.json` yerel
olarak değişmişti; `patchcore.py` ve `run_pipeline.py` değişmedi. Her koşunun
`manifest.json` dosyası 15 kategoriyi, ortak ayarları, seçim yöntemini ve banka
kapasitesini kaydeder. Her `metrics.json` dosyasında 15 kategori bulunur; her
koşuda 15 banka dosyası hedef kapasitededir. AUROC'ların tamamı sonlu ve
0–1 aralığındadır.

Çıktılar `artifacts/bank-size-sweep/<selection>-<N>/` altında `metrics.json`,
`manifest.json`, `run.log` ve `patchcore/<kategori>.pt` olarak saklandı. Veri
ve artifact dosyaları Git'e eklenmedi. Bir koşuyu yeniden üretmek için örnek:

```sh
.venv/bin/python -m anomaly.pipeline.run_pipeline --dataset-root data/mvtec-ad --output-root artifacts/bank-size-sweep/coreset-16384 --image-size 256 --batch-size 8 --projection-dim 256 --seed 42 --device mps --selection coreset --max-patches 16384
```
