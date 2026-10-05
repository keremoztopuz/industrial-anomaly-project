# Claude için teknik devir: API ve model güvenliği düzeltmeleri

Tarih: 5 Ekim 2026. Dal: `fix/api-model-safety`; başlangıç dalı `docs/future-work`. Değişiklikler commit edilmemiş çalışma ağacında. Commit/merge/push ve canlı dağıtım yapılmadı. Önceden var olan `.nodeterm/project.json` ve `.vscode/` kapsam dışı bırakıldı.

## Değişen arayüzler

| Alan | Yeni davranış |
| --- | --- |
| Giriş noktası | `python -m api.main`; test importu `from api.main import app`; eski `api/service.py` kaldırıldı |
| API ayrımı | `main/config/routes/schemas/services`; modeller ve eşikler hâlâ `app.state` üzerinde |
| HTTP | `/`, `/health`, `/categories`, `/model`, `/predict/{category}` ve mevcut başarılı JSON alanları korundu |
| Yanıt şemaları | Pydantic modelleri; nullable `threshold`, `is_anomaly`, `filename`; `400/404/413/422` açıklama ve şemaları |
| Sınırlar | `MAX_UPLOAD_BYTES=10485760`, `MAX_IMAGE_PIXELS=16000000`, `MAX_REQUEST_BYTES=MAX_UPLOAD_BYTES+1048576`; pozitif ortam değişkenleri |
| Başlangıç | Sıfır model hata; olmayan eşik dosyası kabul; bozuk yapı, boolean/string/null/negatif/sonlu olmayan eşik hata |
| Model oluşturma | `--image-size` PatchCore'a geçirilir ve kaydedilir; PatchCore pozitif tamsayı boyut ister |
| Görselleştirme | `--image-size` verilmezse model boyutu; açık uyumsuz değer reddedilir |
| Kayıt | `runs:/<run_id>/model/<uuid>`; aynı run'a tekrar kayıt eski dosyaları değiştirmez |
| Dağıtım | Alias bir kez çözülür; `models:/patchcore-mvtec/<version>` indirilir; ad/boyut/MD5 eşleşmesi zorunlu |
| Drift sonucu | Boolean karar sayısı `decisions`; yetersiz kararda `alarm_rate=null`, `alarm_rate_status=insufficient_data`; sıfır referansta değişim null ve `*_change_status=zero_reference` |

ASGI `RequestSizeLimit`, gövdeyi multipart ayrıştırmasına teslim etmeden önce tamamını sınırlı biçimde okur. Başlıksız parçalı gövde ve gerçek boyuttan küçük `Content-Length` de test edildi. Görüntü okuyucu en fazla dosya sınırı + 1 bayt okur; PNG/JPEG allowlist ve RGB öncesi piksel kontrolü uygular. Pillow'un `DecompressionBombError` durumu 413'e çevrilir.

`load_thresholds` artık `api.services` içindedir; dağıtım da aynı doğrulamayı kullanır. API'de eksik kategori eşiği null karar olarak kalır; dağıtım ise bütün bankalar için kalibrasyon ister. Drift oranı son pencere içindeki gerçek boolean kararlara dayanır; geçmişten ek kayıt çekip pencereyi doldurmaz. Eksik karar, mevcut puan PSI uyarısını/alarmını bastırmaz. Kategoriler arası `overall_status` önceliği korunmuştur; toplam durum tüm kategorilerin eksiksiz olduğunu garanti etmez. CLI null yüzdeleri `n/a` gösterir ve MLflow null ölçümleri atlar.

## Dağıtım akışı ve sınırları

1. Sayısal sürümü sabitle, MLflow paketini indir. En az bir boş olmayan `.pt`, geçerli `thresholds.json` ve tam kategori eşleşmesi iste.
2. Drift etkinse `drift_reference.json`, eşleşen `model_version`/kategoriler ve pozitif pencere iste. Referansı kayıt öncesi hazırla. Eski registry kaynakları yeniden yazılmadı.
3. `gcloud storage objects list ... --raw --format=json` ile adları oku. Hatalar yükselir; yalnızca başarılı boş sonuç yeni yüklemeye izin verir. Listelenen dosyaların **canlı** metadata'sını `objects describe` ile kontrol et.
4. Yeni dosyaları `--if-generation-match=0` ile yükle ve tekrar doğrula. Eksik mevcut sürüm, eksik yükleme, fazladan dosya veya hash/boyut farkında servis/iş güncellemesini durdur.
5. Cloud Run servisini güncelle; drift etkinse mevcut servis imajıyla işi güncelle.

Hash kontrolü GCS'nin base64 MD5 metadata'sını kullanır; dosyalar parça parça hash'lenir. Bu taşıma bütünlüğü kontrolüdür, artifact imzası değildir. Script her `.pt` için tam çıkarım veya her drift referans dizisi için semantik kalibrasyon testi yapmaz. Bucket IAM/retention politikası değiştirilmedi; dışarıdan sonradan yapılan yazmaları engellemez. Sürümlemeli bucket'ta yalnızca eski nesil olarak kalan dosya, canlı `describe` çağrısını başarısız yaparak güvenli biçimde durdurabilir.

Servis ve iş güncellemeleri atomik değildir. İkinci komut hata verirse ilki uygulanmış olabilir; operatör iki kaynağı kontrol etmelidir. Bu çalışmada bütün bulut komutları testlerde taklit edildi; gerçek `deploy_model` çalıştırılmadı.

## Bağımlılıklar ve repo bakımı

- Pillow 12.3.0, pytest 9.0.3. NumPy 2.5.3, HTTPX 0.28.1 ve Pydantic 2.13.5 ilgili doğrudan listelerde açık.
- API ve CI hash kilitleri `uv pip compile` ile yenilendi; yeniden üretme komutları `.in` dosyalarında.
- CI kilidi Flake8 7.4.1 ve pip-audit 2.9.0 içerir. API imajına test/lint araçları eklenmedi.
- `.github/workflows/lamine.yml` → `ci.yml` taşıması yapıldı; ayrıca `permissions: contents: read`, lint ve üç audit komutu eklendi. Henüz commit yok; taşıma ve içerik değişikliği dosya diff'inde incelenebilir.
- `.flake8` yalnızca 99 satır sınırını ayarlar. Gereken mevcut biçim ihlalleri düzeltildi. İş mantığı değişmeyen sekiz kaynak dosyada AST karşılaştırması eşit çıktı.
- `.env`/`.env.*` Git ve Docker bağlamından dışlandı; `.env.example` istisnası test edildi. Otomatik dotenv okuma eklenmedi.

## Çalıştırılan doğrulamalar

```sh
.venv/bin/python -m pytest tests -q
.venv/bin/python -m flake8 anomaly api scripts tests
uv pip check
git diff --check
.venv/bin/python -m pip_audit --no-deps --disable-pip -r requirements-api.lock
.venv/bin/python -m pip_audit --no-deps --disable-pip -r requirements-ci.lock
.venv/bin/python -m pip_audit --no-deps --disable-pip -r requirements.txt
docker build -t industrial-anomaly-api:safety-check .
```

Sonuç: **124 test, 42 alt senaryo geçti**; lint temiz; 186 yerel pakette uyumsuzluk yok. Starlette'in HTTPX test istemcisi ve MLflow'un SQLAlchemy kullanımı için iki deprecation uyarısı var. Bunlar gizlenmedi.

Audit'ler sorgulanabilen bağımlılıklarda bilinen açık bildirmedi. PyPI, `torch==2.14.0+cpu` ve `torchvision==0.29.0+cpu` sürümlerini sorgulayamadığı için kilit taramasında atlar; `requirements.txt` üzerindeki ek tarama temel sürüm advisory'lerini kontrol eder. Yerel API kilidi taraması host platform marker'larını uygular; Linux CI kendi marker'larını uygulayacaktır. Tam Linux x86_64 CI çalışması bu oturumda yürütülmedi.

Docker **Linux ARM64**, Python 3.12 imajı hash kontrolüyle oluşturuldu. Gerçek `artifacts/border-exclusion/coreset-16384-n3-b2/patchcore` klasörü salt okunur bağlandı; 15 model yüklendi. `/health`, `/categories`, `/model`, `/openapi.json` ve gerçek modelle `/predict/bottle` başarılıydı. Beyaz 64×48 PNG için skor `17.34281349182129`, eşik `12.819205284118652` çıktı. Bu yalnızca uçtan uca çalışma kontrolüdür, kalite ölçümü değildir. Modelsiz konteynerin başarısız başladığı doğrulandı. Geçici test konteynerleri kaldırıldı; yerel doğrulama imajı duruyor.

Tekrar Docker kontrolü için README'deki mount komutunu kullan; `/health` ve `/model` çağır, ardından bir PNG ile `/predict/bottle` isteği yap. Model bağlamadan aynı imajı çalıştırmak `No category models` hatasıyla sonlanmalı.

Ek regresyonlar: gerçek geçici MLflow deposunda eski sürüm SHA-256 değişmezliği; 320 CLI → save/load; görselleştirme boyutu; alias değişimi; eksik/farklı uzak dosyalar ve listeleme hatası; eksik/yanlış drift referansı; modelsiz lifespan; geçersiz eşik; upload/piksel/istek sınırları; OpenAPI nullable alanlar; eksik karar ve sıfır referans.

## Mevcut model metadata incelemesi

`torch.load(..., weights_only=True, map_location="cpu")` ile 168 yerel bankanın metadata'sı okundu. 165 eski dosyada `image_size` alanı yok ve mevcut geri uyumluluk varsayımı 256. İki dosyada 320, birinde 384 var. İlişkili manifestinde boyut olan 137 bankada uyuşmazlık yok. Manifesti olmayan kayıtların eğitim boyutu buradan kanıtlanamaz. Model/registry dosyalarında toplu düzeltme yapılmadı.

## Başvurular

- [Pillow 12.3.0 sürüm notları](https://pillow.readthedocs.io/en/stable/releasenotes/12.3.0.html)
- [pytest güvenlik kaydı](https://github.com/advisories/GHSA-6w46-j5rx-g56g)
- [gcloud storage objects list](https://docs.cloud.google.com/sdk/gcloud/reference/storage/objects/list)
- [gcloud storage cp ve generation koşulu](https://docs.cloud.google.com/sdk/gcloud/reference/storage/cp)
