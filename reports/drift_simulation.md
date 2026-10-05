# Drift simülasyonu — 2026-10-05

## Soru

Canlı servisin tahmin günlüğünü okuyan drift kontrolü (`scripts/monitoring/check_drift.py`), gerçek
bir değişikliği yakalıyor ve değişiklik yokken sessiz kalıyor mu?

## Yöntem

Servise gerçek bir hat bağlı olmadığı için hattaki olaylar taklit edildi
(`scripts/monitoring/simulate_drift.py`). Her senaryoda canlı servise `cable` kategorisinden 50 test
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

## Tasarım nasıl değişti

**1. Parlaklık ve kontrast alarm vermiyor.** İlk tasarımda puan, parlaklık, kontrast ve
alarm oranının her biri alarm verebiliyordu. Normal gün senaryosu bile alarm verdi:
parlaklık ve kontrast alarmda, puan uyarıda, alarm oranı normal (%6). Ölçüm doğruydu,
servisin günlüğe yazdığı parlaklık görüntü dosyalarından hesaplananla aynıydı. Sebep,
MVTec'in test görüntülerinin eğitim görüntülerinden farklı bir oturumda çekilmiş olması:
`cable`'da ortalama parlaklık 100,4'ten 101,9'a çıkıyor. Bu fark gözle görülmüyor ve
modelin kararlarını etkilemiyor, ama eğitim görüntüleri çok tekdüze olduğu için PSI bunu
yakalıyor. Bu yüzden durumu yalnızca modelin davranışı belirler: puan dağılımı (PSI) ve
kusurlu denen parçaların oranı.

**2. Otomatik teşhis kaldırıldı.** İkinci tasarım, alarm olduğunda parlaklık veya kontrast
PSI'ı da alarmdaysa "kamerayı ve ışığı kontrol edin", değilse "kusur dalgası olabilir"
diyordu. İlk çalıştırmada dört teşhis de doğru çıktı. Görüntü seçimi değişince kusur
dalgası "kamerayı kontrol edin" teşhisi aldı. Parlaklık ve kontrast normal günde bile
alarmda olduğu için kural neredeyse her zaman "girdi değişti" diyordu; ilk sonuç şanstı.

Bunun yerine ortalamanın ne kadar değiştiğine bakıldı. `cable`'da senaryolar açıkça
ayrışıyor: lamba arızasında parlaklık %40, bulanıklıkta kontrast %6 düşüyor; normal gün
ve kusur dalgasında hiçbiri %2'yi geçmiyor. Ama aynı ölçüm bütün kategorilerde normal gün
için yapıldığında (test kümesinin sağlam görüntüleri ile eğitim görüntüleri):

| Kategori | Parlaklık değişimi | Kontrast değişimi |
| --- | ---: | ---: |
| leather | +%15,1 | +%29,7 |
| grid | −%8,8 | −%18,3 |
| wood | −%1,1 | −%17,1 |
| screw | −%7,5 | −%9,1 |
| hazelnut | +%9,5 | −%6,6 |
| diğer 10 kategori | en fazla %4,5 | en fazla %5,9 |

Bazı kategorilerde hiçbir şey bozulmamışken görülen fark, taklit edilen kamera odağı
sorunundan (%6) büyük. Hiçbir eşik normal günü kamera arızasından bütün kategorilerde
ayıramıyor. Bu yüzden otomatik teşhis kaldırıldı. Rapor, parlaklık ve kontrast
ortalamasının yüzde değişimini sayı olarak gösteriyor; yorumu insan yapıyor.

## Sonuçlar

Son tasarımla, dört senaryo aynı seed ile yeniden çalıştırıldı.

| Senaryo | Kusurlu denen | Puan | Alarm oranı | Durum | Parlaklık | Kontrast |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| Normal gün | 2 / 50 | uyarı | %4 | **uyarı** | +%1,4 | −%1,2 |
| Lamba bozuldu | 11 / 50 | alarm | %22 | **alarm** | −%39,6 | −%40,7 |
| Kamera odağı kaydı | 50 / 50 | alarm | %100 | **alarm** | +%1,4 | −%5,9 |
| Kusur dalgası | 38 / 50 | alarm | %76 | **alarm** | +%1,9 | +%1,2 |

- **Normal gün** alarm vermedi. Puan uyarısı, test görüntülerinin modele eğitimdekilerden
  biraz daha yabancı gelmesinden kaynaklanıyor ([eşik raporunda](threshold_calibration.md)
  `carpet` ve `toothbrush`'ta da görülen eğitim/test farkı).
- **Lamba bozuldu:** Model sağlam parçaların 11'ine kusurlu dedi (normalde 2–3). Hem puan
  dağılımı hem alarm oranı alarm verdi. İlk çalıştırmada alarm oranı %20 ile kendi
  sınırının (%20,7) altında kalmış, alarmı yalnızca puan dağılımı vermişti; iki sinyalin
  birlikte kullanılmasının nedeni bu.
- **Kamera odağı kaydı:** Model bulanık görüntülerin hepsine kusurlu dedi. PatchCore ince
  dokuya bakarak karar veriyor ve bulanıklık bu dokuyu siliyor.
- **Kusur dalgası:** Alarm verdi. Parlaklık ve kontrast değişmediği için okuyan kişi
  sorunun kamerada değil parçalarda olduğunu görebiliyor.

Her kontrol MLflow'daki `monitoring` deneyinde kayıtlıdır: ilk tasarımlarla yapılanlar
`sim-normal`, `sim-dark`, `sim-blur`, `sim-defects`; yeni görüntü seçimiyle yapılanlar
`sim-*-v2` adıyla.

## Sınırlar

- Yalnızca bir kategori (`cable`) ve her senaryo için tek bir 50'lik pencere denendi.
- Bozulmalar yapay: gerçek ışık değişimi renk sıcaklığını ve gölgeleri de değiştirir.
- Referans eğitim görüntülerinden geliyor ve ışığın gün içindeki doğal oynamasını
  içermiyor. Gerçek bir hatta referans, hattın kendi ilk haftalarındaki görüntülerle
  kurulmalıdır. O zaman parlaklık ve kontrast için anlamlı eşikler de belirlenebilir ve
  otomatik teşhis yeniden denenebilir.
