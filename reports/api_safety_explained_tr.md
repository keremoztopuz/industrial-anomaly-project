# Python/FastAPI düzeltmeleri: yeni mezunlar için açıklama

Tarih: 5 Ekim 2026. Çalışma dalı: `fix/api-model-safety`.

Plan uygulandı. HTTP adresleri ve başarılı yanıtların alanları korundu; görüntü yükleme, model kaydı ve dağıtım kontrolleri güçlendirildi. Canlı sisteme dağıtım yapılmadı.

## Görüntü yüklerken ne değişti?

Küçük bir dosya, açıldığında çok büyük bir görüntüye dönüşebilir. Bu yüzden iki ayrı sınır var: dosyanın kapladığı bayt ve görüntünün piksel sayısı. Varsayılanlar 10 MiB ve 16 milyon piksel. İsteğin tamamı da multipart ayrıştırmasından önce sınırlanıyor; başlık göndermeyen veya yanlış `Content-Length` gönderen istemciler bu kontrolü atlayamıyor.

Yalnızca PNG ve JPEG açılıyor. Gri tonlamalı ve RGBA görüntüler model için RGB'ye çevriliyor. Büyük dosya/görüntü `413`, bozuk veya desteklenmeyen görüntü `400` döndürüyor. Reddedilen isteklerde model çalıştırılmıyor. Pillow 12.3.0 ve pytest 9.0.3'e yükseltildi; API ve CI hash kilitleri yeniden üretildi.

## Model sürümleri neden daha güvenilir?

Bir sürüm numarası, hep aynı dosyaları göstermelidir. Eskiden aynı MLflow çalışmasına tekrar kayıt yapmak önceki sürümün dosyalarının üzerine yazabiliyordu. Artık her kayıt `model/<uuid>` altında ayrı bir dizine gidiyor. Geçici gerçek MLflow deposundaki test, ikinci kayıttan sonra birinci modelin SHA-256 özetinin değişmediğini doğruluyor.

Dağıtım, `production` gibi bir alias'ı başta sayısal sürüme çeviriyor. Sonraki indirme bu sayıyı kullanıyor. Uzak depoda bir dosya görülmesi artık yeterli değil: dosya adları, boyutları ve GCS MD5 özetleri yerel paketle eşleşmeli. Eksik veya farklı mevcut paket otomatik olarak onarılmıyor; işlem hata vererek duruyor. Listeleme hatası da “paket yok” sayılmıyor.

Drift işi güncellenecekse referans dosyası bulunmalı; sürümü, kategori listesi ve pencere ayarı uygun olmalı. Bu kontroller geçmeden servis/iş güncellenmiyor.

## Görüntü boyutu ve başlangıç neden önemli?

320 piksel görüntülerle hazırlanan modelin kendisini 256 piksel olarak tanıtması yanlış ön işleme yol açabilir. Model oluşturma komutu artık seçilen boyutu modele aktarıyor. Test, CLI'deki 320 değerinin kaydedilen ve tekrar yüklenen modele ulaştığını kontrol ediyor. Görselleştirme boyutu modelden alıyor; uyumsuz açık seçenek hata veriyor.

Yerelde 168 model dosyası salt okunur incelendi. 165 dosya eski 256 varsayımını kullanıyor; iki dosyada 320, bir dosyada 384 metadata'sı var. Boyut bilgisi bulunan 137 eğitim manifestiyle karşılaştırmada uyuşmazlık görülmedi. Bu, kaydı olmayan eski eğitimlerin boyutunu kanıtlamaz; hiçbir model dosyası değiştirilmedi.

Model dizini boşsa uygulama artık açılışta açıklayıcı hata veriyor. Eşik dosyası yoksa mevcut `threshold: null`, `is_anomaly: null` davranışı korunuyor. Dosya var ama geçersizse açılış duruyor. `/health`, başarıyla başlayan HTTP sürecinin yaşadığını bildiriyor.

## Drift kontrolünde “bilmiyorum” artık nasıl gösteriliyor?

`is_anomaly: null`, ürünün sağlam olduğu anlamına gelmez; karar için eşik bulunmadığını gösterir. Alarm oranı yalnızca gerçek `true`/`false` kararlarından hesaplanıyor. Pencereyi dolduracak karar yoksa oran `null`, ilgili durum `insufficient_data` oluyor. Puan dağılımında görülen uyarı/alarm yine korunuyor.

Parlaklık veya kontrast referansı sıfırsa yüzde değişim hesaplanamıyor. Bölme hatası yerine `null` ve `zero_reference` dönüyor. Komut satırı bunu `n/a` gösteriyor; MLflow'a sayısal ölçüm olarak göndermiyor.

## Dosyalar ve geliştirme akışı

`api/main.py` uygulamayı, `config.py` ayarları, `routes.py` HTTP işlemlerini, `schemas.py` yanıt türlerini, `services.py` yükleme/tahmin işlerini içeriyor. Yeni komut `python -m api.main`. `/docs` üzerinden yanıt türleri, boş olabilen alanlar ve hata kodları görülebiliyor.

`.env` dosyaları Git ve Docker kapsamı dışında; `.env.example` örnek olarak tutuluyor. Uygulama `.env` dosyasını kendiliğinden okumuyor: değişkenler dışarıdan aktarılmalı. İş akışı `ci.yml` adını aldı; okuma yetkisi, lint ve bağımlılık taraması eklendi. Flake8 satır sınırı 99; ihlal türleri gizlenmedi.

## Doğrulama sonucu

- **124 test ve 42 alt senaryo geçti.** Önceki davranış kontrolleri korundu, dağıtım testleri yeni güvenli davranışa uyarlandı.
- Flake8 temiz; `uv pip check` 186 yerel pakette uyumsuzluk bildirmedi.
- İki kilit ve doğrudan bağımlılıklar tarandı; sorgulanabilen paketler için bilinen açık bildirilmedi. `+cpu` paketleri PyPI sorgusunda atlandığı için Torch/Torchvision'ın temel sürümleri ayrıca tarandı.
- Linux ARM64 Docker imajı hash doğrulamalı API kilidiyle oluşturuldu. Gerçek 15 model yüklendi; `/health`, `/categories`, `/model`, OpenAPI ve bir PNG tahmini başarılı oldu. Boş model dizininde başlangıcın başarısız olduğu da doğrulandı.
- İki bağımlılık kaynaklı kullanım dışı bırakma uyarısı kaldı: Starlette/HTTPX ve MLflow/SQLAlchemy.

Bu kontroller canlı Cloud Run, IAM veya tüm modellerin tahmin kalitesi için bir doğrulama değildir. Servis ve drift işi bulutta ayrı komutlarla güncellenir; tek bir işlem gibi geri alınmaz. Bu sınırlamalar teknik devir raporunda açıklanıyor.

Git geçmişine commit, merge veya push yapılmadı. Başlangıçtaki `docs/future-work` dalından ayrı çalışma dalı açıldı; mevcut `.nodeterm/project.json` ve `.vscode/` kullanıcı değişikliklerine dokunulmadı.
