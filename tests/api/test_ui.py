"""The test page ships inside the API, including its sample images."""

import unittest
from collections import defaultdict
from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from tests.support.api_fakes import FakeModel, make_client


class TestPageTests(unittest.TestCase):
    def setUp(self):
        self.client = make_client(
            {"bottle": FakeModel(0.5)}, thresholds={"bottle": 0.4}
        )

    def test_page_redirect_and_assets(self):
        redirect = self.client.get("/ui", follow_redirects=False)
        self.assertEqual(redirect.status_code, 307)
        self.assertEqual(redirect.headers["location"], "./ui/")
        for path, content_type in (
            ("/ui/", "text/html"),
            ("/ui/style.css", "text/css"),
            ("/ui/app.js", "javascript"),
            ("/ui/samples.json", "application/json"),
            ("/ui/samples/LICENSE.txt", "text/plain"),
        ):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn(content_type, response.headers["content-type"])
        page = self.client.get("/ui/").text
        self.assertIn('lang="en"', page)
        self.assertIn('href="style.css"', page)
        self.assertIn('src="app.js"', page)
        self.assertIn('href="../docs"', page)

    def test_redirect_preserves_https_and_proxy_path_prefix(self):
        client = TestClient(
            self.client.app,
            base_url="https://public.example/proxy",
            root_path="/proxy",
        )
        response = client.get("/ui")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(str(response.url), "https://public.example/proxy/ui/")
        self.assertEqual(response.history[0].headers["location"], "./ui/")

    def test_existing_routes_keep_their_contracts(self):
        self.assertEqual(
            self.client.get("/").json(), {"status": "API is running"}
        )
        self.assertEqual(
            self.client.get("/health").json(), {"status": "healthy"}
        )
        self.assertEqual(
            self.client.get("/categories").json(), {"categories": ["bottle"]}
        )
        self.assertEqual(self.client.get("/docs").status_code, 200)

    def test_manifest_has_two_valid_original_images_per_category(self):
        manifest = self.client.get("/ui/samples.json").json()
        self.assertEqual(manifest["license"], "CC BY-NC-SA 4.0")
        self.assertEqual(len(manifest["samples"]), 30)
        categories = defaultdict(set)
        files = set()
        for sample in manifest["samples"]:
            with self.subTest(sample=sample["file"]):
                categories[sample["category"]].add(sample["kind"])
                self.assertNotIn(sample["file"], files)
                files.add(sample["file"])
                self.assertTrue(sample["source_path"].startswith("data/"))
                self.assertEqual(
                    sample["label"] == "good", sample["kind"] == "normal"
                )
                response = self.client.get(f'/ui/{sample["file"]}')
                self.assertEqual(response.status_code, 200)
                self.assertIn("image/png", response.headers["content-type"])
                self.assertLess(len(response.content), 10 * 1024 * 1024)
                with Image.open(BytesIO(response.content)) as image:
                    self.assertEqual(image.format, "PNG")
                    self.assertLessEqual(image.width * image.height, 16000000)
                    image.verify()
        self.assertEqual(len(categories), 15)
        for kinds in categories.values():
            self.assertEqual(kinds, {"normal", "defective"})

    def test_packaged_samples_use_the_existing_upload_endpoint(self):
        for kind in ("normal", "defective"):
            with self.subTest(kind=kind):
                filename = f"bottle-{kind}.png"
                image = self.client.get(f"/ui/samples/{filename}").content
                response = self.client.post(
                    "/predict/bottle",
                    files={"upload_file": (filename, image, "image/png")},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    response.json(),
                    {
                        "category": "bottle",
                        "filename": filename,
                        "anomaly_score": 0.5,
                        "threshold": 0.4,
                        "is_anomaly": True,
                    },
                )

    def test_static_files_do_not_expose_api_source_or_missing_paths(self):
        for path in (
            "/ui/missing.png",
            "/ui/%2e%2e%2fmain.py",
            "/ui/%2e%2e%2f%2e%2e%2f.env",
        ):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 404)
