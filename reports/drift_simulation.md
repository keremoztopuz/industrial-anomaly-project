# Drift simülasyonu — 2026-10-05

## Soru

Canlı servisin tahmin günlüğünü okuyan drift kontrolü (`scripts/check_drift.py`), gerçek
bir değişikliği yakalıyor ve değişiklik yokken sessiz kalıyor mu? Değişikliğin sebebini
doğru söyleyebiliyor mu?

## Yöntem

Servise gerçek bir hat bağlı olmadığı için hattaki olaylar taklit edildi
(`scripts/simulate_drift.py`). Her senaryoda canlı servise `cable` kategorisinden 50 test
görüntüsü gönderildi, ardından drift kontrolü çalıştırıldı. `cable`, test kümesinde 50'den
fazla sağlam görüntüsü olan iki kategoriden biri ve senaryolar için yeterli kusurlu
görüntüsü var. Eğitim görüntüleri kullanılmadı, çünkü canlı banka onları içerdiği için
puanları yapay olarak düşük çıkar.

| Senaryo | Gönderilen |
| --- | --- |
| Normal gün | 50 sağlam görüntü, değişiklik yok |
| Lamba bozuldu | Aynı sağlam görüntüler, parlaklık %60'a indirilmiş |
| Kamera odağı kaydı | Aynı sağlam görüntüler, Gauss bulanıklığı (yarıçap 4) |
| Kusur dalgası | 40 kusurlu ve 10 sağlam görüntü, normal ışık |

## İlk sonuç ve düzeltme

İlk tasarımda dört sinyalin (puan, parlaklık, kontrast, alarm oranı) hepsi alarm
verebiliyordu. **Normal gün** senaryosu bile alarm verdi: parlaklık ve kontrast alarmda,
puan uyarıda, alarm oranı normal (%6).

Ölçüm doğruydu: servisin günlüğe yazdığı parlaklık ile görüntü dosyalarından hesaplanan
parlaklık aynıydı (ortalama 101,9). Sebep, MVTec'in test görüntülerinin eğitim
görüntülerinden biraz farklı çekilmiş olması. `cable`'da ortalama parlaklık 100,4'ten
101,9'a çıkıyor. 255'lik ölçekte bu gözle görülmeyecek kadar küçük bir fark. Ama eğitim
görüntüleri birbirine çok benzediği için PSI bunu yakalıyor. Test kümesinin sağlam
görüntüleri eğitim referansıyla karşılaştırıldığında parlaklık veya kontrast PSI'ı
15 kategorinin 13'ünde alarm eşiğini aşıyor.

Bu fark modelin davranışını bozmuyor (`cable` alarm oranı %6). Gerçek bir hatta da ışık gün
içinde bu kadar oynar ve böyle bir monitör her gün alarm verirdi. Bu yüzden karar şu oldu:
**durumu yalnızca modelin davranışı belirler** (puan PSI'ı ve alarm oranı). Parlaklık ve
kontrast tek başına alarm vermez. Bir alarm olduğunda sebebi açıklar:

- Puan veya alarm oranı alarmda **ve** parlaklık ya da kontrast alarmda: "girdi
  görüntüleri değişti, kamerayı ve ışığı kontrol edin".
- Alarm oranı alarmda, girdiler değişmemiş: "ışık aynıyken daha çok parça işaretleniyor,
  gerçek bir kusur dalgası olabilir".

## Sonuçlar (düzeltmeden sonra)

| Senaryo | Kusurlu denen | Puan | Parlaklık | Kontrast | Alarm oranı | Durum | Teşhis |
| --- | ---: | --- | --- | --- | ---: | --- | --- |
| Normal gün | 3 / 50 | uyarı | alarm | alarm | %6 | **uyarı** | girdi değişti |
| Lamba bozuldu | 10 / 50 | alarm | alarm | alarm | %20 | **alarm** | girdi değişti |
| Kamera odağı kaydı | 50 / 50 | alarm | alarm | alarm | %100 | **alarm** | girdi değişti |
| Kusur dalgası | 37 / 50 | alarm | uyarı | uyarı | %74 | **alarm** | kusur dalgası olabilir |

- **Normal gün** alarm vermedi. Puan uyarısı, test görüntülerinin modele eğitimdekilerden
  biraz daha yabancı gelmesinden kaynaklanıyor ([eşik raporunda](threshold_calibration.md)
  `carpet` ve `toothbrush`'ta da görülen eğitim/test farkı).
- **Lamba bozuldu:** Model sağlam parçaların 10'una kusurlu demeye başladı (normalde 3).
  Alarm oranı (%20) kendi sınırının (%20,7) hemen altında kaldı. Puan dağılımı ise alarm
  verdi. İki sinyalin birlikte kullanılmasının nedeni bu.
- **Kamera odağı kaydı:** Model bulanık görüntülerin hepsine kusurlu dedi. PatchCore ince
  dokuya bakarak karar veriyor ve bulanıklık bu dokuyu siliyor.
- **Kusur dalgası:** Alarm verdi ve sebebi kamera yerine kusurlar olarak doğru teşhis etti.

Her kontrol MLflow'daki `monitoring` deneyinde `sim-*` adıyla kayıtlıdır. İlk tasarımla
yapılan normal gün kontrolü de orada `sim-normal` adıyla ve `alarm` durumuyla duruyor.

## Sınırlar

- Yalnızca bir kategori (`cable`) ve her senaryo için tek bir 50'lik pencere denendi.
- Bozulmalar yapay: gerçek ışık değişimi renk sıcaklığını ve gölgeleri de değiştirir.
- Normal gün senaryosunun uyarı vermesi, eğitim ve test görüntüleri arasındaki farkın
  puanlara da az da olsa yansıdığını gösteriyor. Gerçek bir hatta referans, hattın kendi
  ilk haftalarındaki görüntülerle güncellenmelidir.
