# Patch seçimi deneyi — 2026-10-01

## Ne değişti?

ResNet ve görüntü işleme aynı kaldı. Normal görüntülerden çıkan patch özelliklerini
saklama yöntemi değişti:

- **Rastgele:** Bütün normal patch'lerden 2.048 tanesi seçilir.
- **Coreset:** Önce rastgele 8.192 aday alınır. İlk adaydan başlayarak mevcut
  seçilenlere en uzak aday tekrar tekrar eklenir; bankada 2.048 patch kalır.

Basit örnek: Normal özellikler `0, 1, 5, 10` ve banka kapasitesi iki olsun.
İlk seçim `0` ise coreset ikinci olarak en uzak `10` değerini alır. Rastgele
seçim `0, 1` de alabilirdi. Test özelliği `9` için en yakın normal değere
uzaklık ilk bankada `1`, ikinci bankada `8` olur. Gerçekte özellikler tek sayı
değil, 256 boyutlu vektörlerdir. Modelin anomali puanı bu en yakın uzaklıklardan
üretilir; bu deneyde loss, optimizer veya ResNet eğitimi yoktur.

## Deney ayarları

İki yöntem aynı MVTec AD `samples.json` dosyasını kullandı
(`SHA256: 4382cbf3b69e515b808d2e12232a84df042ed067243ba304d5f6a820e22375e0`).
Görüntü boyutu 256 × 256, batch 8, banka 2.048 patch, izdüşüm 256 boyut,
seed 42 ve cihaz Apple MPS idi. Rastgele yöntem de yeni kodla tekrar çalıştırıldı;
önceki baseline sonuçlarıyla aynı değerleri verdi. Altı koşunun manifestindeki
kaynak commit `9da92dc12d7505b564f522d1f110f0953bd7ba7d`, `git_dirty=false`.

## Sonuçlar

AUROC'ta büyük değer daha iyi; fark = coreset − rastgele. Image AUROC görüntünün
normal/anormal ayrımını, pixel AUROC kusurun yerini ayırt etmeyi ölçer.

| Kategori | Rastgele image | Coreset image | Fark | Rastgele pixel | Coreset pixel | Fark |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| screw | 0.5919 | 0.6300 | +0.0381 | 0.9687 | 0.9777 | +0.0090 |
| grid | 0.6850 | 0.6667 | −0.0184 | 0.9197 | 0.9518 | +0.0321 |
| transistor | 0.6804 | 0.8075 | +0.1271 | 0.7865 | 0.8352 | +0.0488 |
| **Üç kategori ortalaması** | **0.6525** | **0.7014** | **+0.0489** | **0.8916** | **0.9216** | **+0.0300** |

Coreset bu üç kategorinin ortalamasını iyileştirdi; fakat `grid` image AUROC'u
düştü. Pixel ayrımı iyileşirken görüntü ayrımının düşmesi mümkündür: görüntü
puanı patch uzaklıklarının en büyüğünden gelir. Bu deney tek seed ve yalnızca
üç kategori içeriyor; bütün veri setinde üstünlük kanıtı değildir.

Her koşunun `metrics.json`, `manifest.json` ve `patchcore/<kategori>.pt` dosyaları
`artifacts/coreset-experiment/<yöntem>/<kategori>/` altında üretildi. Veri ve
model dosyaları Git'e eklenmedi.
