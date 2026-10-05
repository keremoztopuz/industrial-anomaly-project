# Yerel komşuluk birleştirme — 2026-10-01

## Ne değişti?

PatchCore, her patch'i tek başına değil, çevresindeki patch'lerle birlikte temsil eder.
Bu dalda `neighborhood` ayarı eklendi. `neighborhood=3` olduğunda `layer2` ve `layer3`
özellik haritalarının her noktası, kendisi ve 8 komşusundan oluşan 3 × 3 pencerenin
ortalamasıyla değiştirilir. Kenarlarda yalnızca harita içindeki komşular ortalamaya girer.
Birleştirme hem banka oluşturulurken hem de test sırasında uygulanır. Varsayılan değer
`1` olduğundan, ayar verilmezse önceki davranış değişmez.

Basit örnek: Tek bir patch'te yanlışlıkla parlak bir yansıma varsa, onun özelliği
komşularından çok farklıdır ve tek başına yüksek puan alabilir. 3 × 3 ortalama bu tek
noktalık sapmayı yumuşatır. Birkaç komşu patch'i kaplayan gerçek bir çizik ise ortalamada
da belirgin kalır.

## Deney ayarları

Bütün koşular MVTec AD'nin 15 kategorisinde, aynı `samples.json` dosyasıyla yapıldı
(`SHA256: 4382cbf3b69e515b808d2e12232a84df042ed067243ba304d5f6a820e22375e0`).
Görüntü 256 × 256, batch 8, banka 16.384 patch, izdüşüm 256 boyut, seed 42, cihaz
Apple MPS. `neighborhood=1` sütunları [banka boyutu taramasından](bank_size_sweep.md)
alındı; `neighborhood=3` koşuları bu dalda yapıldı. İki yeni manifestte kaynak commit
`01a4031e8533839cfb490cf84288cb8aba9762a2`, `git_dirty=false`. Dal daha sonra `main`
üzerine taşındı; Python dosyalarında fark yoktur, yalnızca rapor dosyaları eklenmiştir.

AUROC'ta büyük değer daha iyi; fark = `neighborhood=3` − `neighborhood=1`. Image AUROC
görüntünün normal/anormal ayrımını, pixel AUROC kusurun yerini ayırt etmeyi ölçer.

## Sonuçlar

### Image AUROC

| Kategori | Rastgele, 1 | Rastgele, 3 | Fark | Coreset, 1 | Coreset, 3 | Fark |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bottle | 0.9897 | 1.0000 | +0.0103 | 0.9968 | 1.0000 | +0.0032 |
| cable | 0.8371 | 0.9010 | +0.0639 | 0.8896 | 0.9192 | +0.0296 |
| capsule | 0.9047 | 0.9402 | +0.0355 | 0.9254 | 0.9302 | +0.0048 |
| carpet | 0.9498 | 0.9310 | −0.0189 | 0.9623 | 0.9342 | −0.0281 |
| grid | 0.5881 | 0.6115 | +0.0234 | 0.7243 | 0.6834 | −0.0409 |
| hazelnut | 0.9689 | 0.9996 | +0.0307 | 0.9954 | 1.0000 | +0.0046 |
| leather | 0.9986 | 1.0000 | +0.0014 | 1.0000 | 0.9997 | −0.0003 |
| metal_nut | 0.9765 | 0.9936 | +0.0171 | 0.9897 | 0.9985 | +0.0088 |
| pill | 0.9135 | 0.9446 | +0.0311 | 0.9285 | 0.9433 | +0.0147 |
| screw | 0.7920 | 0.7661 | −0.0258 | 0.9377 | 0.9010 | −0.0367 |
| tile | 0.8990 | 0.9509 | +0.0519 | 0.9098 | 0.9574 | +0.0476 |
| toothbrush | 0.8556 | 0.8667 | +0.0111 | 0.8667 | 0.8528 | −0.0139 |
| transistor | 0.8642 | 0.8496 | −0.0146 | 0.9117 | 0.9071 | −0.0046 |
| wood | 0.9798 | 0.9702 | −0.0096 | 0.9939 | 0.9719 | −0.0219 |
| zipper | 0.9262 | 0.9396 | +0.0134 | 0.9443 | 0.9493 | +0.0050 |
| **Makro ortalama** | **0.8963** | **0.9110** | **+0.0147** | **0.9317** | **0.9299** | **−0.0019** |

### Pixel AUROC

| Kategori | Rastgele, 1 | Rastgele, 3 | Fark | Coreset, 1 | Coreset, 3 | Fark |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bottle | 0.9833 | 0.9871 | +0.0038 | 0.9852 | 0.9872 | +0.0020 |
| cable | 0.9536 | 0.9791 | +0.0255 | 0.9667 | 0.9820 | +0.0153 |
| capsule | 0.9792 | 0.9862 | +0.0069 | 0.9840 | 0.9876 | +0.0037 |
| carpet | 0.9835 | 0.9857 | +0.0021 | 0.9870 | 0.9862 | −0.0009 |
| grid | 0.9532 | 0.9462 | −0.0070 | 0.9768 | 0.9613 | −0.0155 |
| hazelnut | 0.9842 | 0.9879 | +0.0037 | 0.9865 | 0.9883 | +0.0018 |
| leather | 0.9925 | 0.9919 | −0.0006 | 0.9929 | 0.9921 | −0.0009 |
| metal_nut | 0.9597 | 0.9820 | +0.0223 | 0.9717 | 0.9846 | +0.0129 |
| pill | 0.9338 | 0.9741 | +0.0403 | 0.9517 | 0.9812 | +0.0295 |
| screw | 0.9830 | 0.9809 | −0.0021 | 0.9886 | 0.9868 | −0.0018 |
| tile | 0.9211 | 0.9534 | +0.0323 | 0.9393 | 0.9550 | +0.0157 |
| toothbrush | 0.9885 | 0.9875 | −0.0010 | 0.9883 | 0.9871 | −0.0012 |
| transistor | 0.8403 | 0.9399 | +0.0996 | 0.8740 | 0.9561 | +0.0822 |
| wood | 0.9325 | 0.9337 | +0.0012 | 0.9395 | 0.9367 | −0.0028 |
| zipper | 0.9744 | 0.9741 | −0.0003 | 0.9820 | 0.9782 | −0.0038 |
| **Makro ortalama** | **0.9575** | **0.9726** | **+0.0151** | **0.9676** | **0.9767** | **+0.0091** |

Pixel AUROC iki seçim yönteminde de arttı. En büyük artış `transistor`'da oldu
(coreset 0.8740 → 0.9563). Image AUROC rastgele seçimde arttı, coreset'te ise
değişmedi. `cable`, `pill` ve `tile` iyileşirken `grid`, `screw`, `carpet` ve
`wood` düştü. Bu tabloya göre en iyi image AUROC coreset ve `neighborhood=1` ile
(0.9317), en iyi pixel AUROC ise coreset ve `neighborhood=3` ile (0.9767) elde edildi.

## Küçük banka denemesi

Aynı ayar önce 2.048 patch'lik bankayla `screw`, `grid` ve `transistor` kategorilerinde
denendi. Bu koşularda kaynak commit de `01a4031`, `git_dirty=false` idi.

| Seçim | Image AUROC, 1 → 3 | Pixel AUROC, 1 → 3 |
| --- | ---: | ---: |
| Rastgele | 0.6525 → 0.5748 | 0.8916 → 0.9309 |
| Coreset | 0.7014 → 0.6918 | 0.9216 → 0.9540 |

Küçük bankada pixel AUROC yine arttı, fakat image AUROC düştü; `grid` rastgele
seçimde 0.4879'a indi. 16.384 patch'te bu düşüş büyük ölçüde kayboldu. Bunun nedeni
ayrıca test edilmedi.

## Sınırlar ve süre

Sonuçlar tek seed'e dayanıyor; belirsizlik aralığı yoktur. Süreler `time -p` ile
ölçüldü: 16.384 rastgele 222,58 sn, 16.384 coreset 2.454,70 sn. Bu süreler, aynı
makinede farklı zamanlarda yapılan `neighborhood=1` koşularıyla doğrudan
karşılaştırılmamalıdır. 2.048 patch'lik denemeler başka bir koşuyla aynı anda
çalıştığından onların süreleri raporlanmadı.

Çıktılar `artifacts/local-aggregation-16384/<seçim>/` altında `metrics.json`,
`manifest.json`, `run.log` ve `patchcore/<kategori>.pt` olarak üretildi. Veri ve
model dosyaları Git'e eklenmedi. Yeniden üretmek için:

```sh
.venv/bin/python -m anomaly.pipeline.run_pipeline --dataset-root data/mvtec-ad --output-root artifacts/local-aggregation-16384/coreset --image-size 256 --batch-size 8 --projection-dim 256 --seed 42 --device mps --max-patches 16384 --selection coreset --neighborhood 3
```
