# Kenar patch'lerini görüntü puanından çıkarma — 2026-10-02

## Sorun

`grid` kategorisinde image AUROC, banka büyüdükçe veya özellikler iyileştikçe düzelmiyordu
(coreset, 16.384 patch: 0.7243). Bunun nedenini bulmak için her test görüntüsünde
görüntü puanını veren patch'in yerine bakıldı. Görüntü puanı, bütün patch'lerin
en yüksek uzaklığıdır.

`grid`'de 21 sağlam test görüntüsünün 21'inde de en yüksek uzaklık görüntünün en dış
patch halkasındaydı; çoğu alt köşelerdeydi. 57 kusurlu görüntünün 23'ünde de puan
kusurdan değil kenardan geliyordu. Yani görüntü puanını çoğunlukla kusur değil köşe
belirliyordu.

## Neden?

Önce bankanın küçük olması ihtimali denendi. `grid`'in bütün sağlam patch'leri
(270.336 adet) seçim yapılmadan bankaya kondu. Image AUROC 0.8613'e çıktı, ama
sağlam görüntülerin 21'inde de en yüksek uzaklık hâlâ kenardaydı. Sağlam test
görüntülerinde ortalama en yakın komşu uzaklığı iç patch'lerde 14,32, kenarlarda
19,48, köşelerde 27,22 oldu.

Kenar patch'leri bankada az olduğu için değil, kendileri farklı olduğu için uzak çıkıyor.
Ağ, görüntünün dışını sıfırla doldurarak (zero padding) kenarlardaki özellikleri
hesaplar. Bu özellikler görüntü içeriğinden çok bu dolgunun etkisini taşır ve sağlam
görüntüler arasında bile değişkendir.

## Değişiklik

`PatchCore` ve iki komut satırı aracına `border` ayarı eklendi. `border=N`, görüntü
puanını hesaplarken en dış N patch halkasını yok sayar. Anomali haritası bütün patch'leri
korur; bu yüzden pixel AUROC değişmez. Varsayılan `0` olduğundan eski davranış aynı
kalır. Ayar banka dosyasına kaydedilir; ayarı olmayan eski dosyalar `0` ile açılır.

Basit örnek: 256 × 256 görüntüde patch ızgarası 32 × 32'dir ve bir patch yaklaşık
8 piksele karşılık gelir. `border=2` ile görüntü puanı kenardan yaklaşık 16 piksel
içerideki 28 × 28 patch'ten hesaplanır.

## Bütün kategorilerde etkisi

Aşağıdaki tablo, mevcut coreset/16.384 bankalarının test görüntülerinde farklı görüntü
puanlarıyla yeniden puanlanmasından elde edildi. Bankalar yeniden oluşturulmadı.

### Image AUROC

| Kategori | n=1, kenar yok | n=1, border 1 | n=1, border 2 | n=3, kenar yok | n=3, border 1 | n=3, border 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bottle | 0.9968 | 0.9968 | 0.9968 | 1.0000 | 1.0000 | 1.0000 |
| cable | 0.8896 | 0.9440 | 0.9442 | 0.9192 | 0.9880 | 0.9921 |
| capsule | 0.9254 | 0.9753 | 0.9753 | 0.9302 | 0.9741 | 0.9733 |
| carpet | 0.9623 | 0.9940 | 0.9952 | 0.9342 | 0.9647 | 0.9819 |
| grid | 0.7243 | 0.9925 | 0.9958 | 0.6834 | 0.9106 | 0.9758 |
| hazelnut | 0.9954 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| leather | 1.0000 | 1.0000 | 1.0000 | 0.9997 | 1.0000 | 1.0000 |
| metal_nut | 0.9897 | 0.9897 | 0.9897 | 0.9985 | 0.9985 | 0.9985 |
| pill | 0.9285 | 0.9247 | 0.9247 | 0.9433 | 0.9444 | 0.9444 |
| screw | 0.9377 | 0.9508 | 0.9508 | 0.9010 | 0.9102 | 0.9250 |
| tile | 0.9098 | 0.9758 | 0.9863 | 0.9574 | 0.9971 | 0.9996 |
| toothbrush | 0.8667 | 0.9167 | 0.9778 | 0.8528 | 0.8861 | 0.9194 |
| transistor | 0.9117 | 0.9333 | 0.9342 | 0.9071 | 0.9767 | 0.9954 |
| wood | 0.9939 | 0.9939 | 0.9939 | 0.9719 | 0.9842 | 0.9868 |
| zipper | 0.9443 | 0.9572 | 0.9666 | 0.9493 | 0.9543 | 0.9632 |
| **Makro ortalama** | **0.9317** | **0.9696** | **0.9754** | **0.9299** | **0.9659** | **0.9770** |

`n` = `neighborhood`. Kenarı dışlamak 15 kategorinin 14'ünde image AUROC'u düşürmedi.
`neighborhood=1` ile `pill` 0.9285'ten 0.9247'ye indi (`border` 1 ve 2'de). Bu düşüş
`neighborhood=3` ile görülmedi. `neighborhood=3`, kenar etkisini bir halka
daha içeri taşıdığı için `border=2` ile `border=1`'den belirgin şekilde daha iyi sonuç verdi
(`grid` 0.9106 → 0.9758).

## Resmi koşu

Pipeline, `neighborhood=3` ve `border=2` ile 15 kategoride baştan çalıştırıldı.
Ayarlar: coreset, 16.384 patch, görüntü 256 × 256, batch 8, izdüşüm 256 boyut, seed 42,
Apple MPS. Karşılaştırma sütunu [yerel komşuluk raporundaki](local_aggregation.md)
`neighborhood=3`, `border=0` koşusudur.

| Kategori | Image, border 0 | Image, border 2 | Fark | Pixel AUROC |
| --- | ---: | ---: | ---: | ---: |
| bottle | 1.0000 | 1.0000 | +0.0000 | 0.9872 |
| cable | 0.9192 | 0.9921 | +0.0729 | 0.9820 |
| capsule | 0.9302 | 0.9733 | +0.0431 | 0.9876 |
| carpet | 0.9342 | 0.9819 | +0.0478 | 0.9862 |
| grid | 0.6834 | 0.9758 | +0.2924 | 0.9613 |
| hazelnut | 1.0000 | 1.0000 | +0.0000 | 0.9883 |
| leather | 0.9997 | 1.0000 | +0.0003 | 0.9921 |
| metal_nut | 0.9985 | 0.9985 | +0.0000 | 0.9846 |
| pill | 0.9433 | 0.9444 | +0.0011 | 0.9812 |
| screw | 0.9010 | 0.9250 | +0.0240 | 0.9868 |
| tile | 0.9574 | 0.9996 | +0.0422 | 0.9550 |
| toothbrush | 0.8528 | 0.9194 | +0.0667 | 0.9871 |
| transistor | 0.9071 | 0.9954 | +0.0883 | 0.9561 |
| wood | 0.9719 | 0.9868 | +0.0149 | 0.9367 |
| zipper | 0.9493 | 0.9632 | +0.0139 | 0.9782 |
| **Makro ortalama** | **0.9299** | **0.9770** | **+0.0472** | **0.9767** |

Koşunun image AUROC değerleri, yukarıdaki yeniden puanlamayla bütün kategorilerde aynı
çıktı. Pixel AUROC değerleri `border=0` koşusuyla aynıdır. Süre `time -p` ile 3.205,49 sn.

Manifestte kaynak commit `967b7ebda16590a7688296da9120ca88e6af5ad5`, veri `samples.json`
SHA256 değeri `4382cbf3b69e515b808d2e12232a84df042ed067243ba304d5f6a820e22375e0`.
`git_dirty=true`: koşu sırasında yalnızca nodeterm'in `.nodeterm/` dosyaları yerel olarak
değişmişti; Python dosyaları commit ile aynıydı.

## Sınırlar

- Kenarın etkisi ve `border` değeri test sonuçlarına bakılarak bulundu. MVTec AD'de ayrı
  bir doğrulama kümesi yoktur; bu yüzden bu sonuç test kümesine bir miktar uyum içerebilir.
  Gerçek bir hatta `border`, test verisinden önce, sağlam görüntülerden ayrılmış bir
  doğrulama kümesiyle seçilmelidir.
- Yalnızca dışlanan şeritte kalan bir kusur (kenardan yaklaşık 16 piksel) görüntü
  puanını etkilemez. Anomali haritasında yine görünür.
- Sonuçlar tek seed'e dayanıyor; belirsizlik aralığı yoktur.

Çıktılar `artifacts/border-exclusion/coreset-16384-n3-b2/` altında üretildi. Veri ve
model dosyaları Git'e eklenmedi. Yeniden üretmek için:

```sh
.venv/bin/python -m anomaly.pipeline.run_pipeline --dataset-root data/mvtec-ad --output-root artifacts/border-exclusion/coreset-16384-n3-b2 --image-size 256 --batch-size 8 --projection-dim 256 --seed 42 --device mps --max-patches 16384 --selection coreset --neighborhood 3 --border 2
```
