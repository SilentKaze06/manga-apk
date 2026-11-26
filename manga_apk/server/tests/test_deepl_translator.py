import os
import unittest
from unittest.mock import patch, MagicMock
from deepl_translator import is_deepl_supported, translate_dialogues_with_deepl, translate_single_with_deepl


class TestDeepLTranslator(unittest.TestCase):
    def setUp(self):
        self.env_patch = patch.dict(os.environ, {"DEEPL_API_KEY": "", "DEEPL_API_KEY_FILE": ""}, clear=False)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)

    def test_unsupported_uzbek_language(self):
        # DeepL does not support Uzbek
        self.assertFalse(is_deepl_supported("uz"))
        self.assertFalse(is_deepl_supported("uz-UZ"))
        # When Uzbek is requested, DeepL must cleanly return None so current fallback is used
        res = translate_dialogues_with_deepl(["Salom"], target_lang="uz", api_key="fake-key")
        self.assertIsNone(res)

    def test_supported_russian_language(self):
        # DeepL supports Russian, English, Japanese, etc.
        self.assertTrue(is_deepl_supported("ru"))
        self.assertTrue(is_deepl_supported("en"))
        self.assertTrue(is_deepl_supported("ja"))

    def test_missing_api_key_returns_none(self):
        with patch.dict(os.environ, {"DEEPL_API_KEY": "", "DEEPL_API_KEY_FILE": ""}, clear=False):
            res = translate_dialogues_with_deepl(["Hello"], target_lang="ru", api_key="")
            self.assertIsNone(res)

    @patch("urllib.request.urlopen")
    def test_successful_batch_translation(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"translations": [{"text": "\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82"}, {"text": "\xd0\x9c\xd0\xb8\xd1\x80"}]}'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        res = translate_dialogues_with_deepl(["Hello", "World"], target_lang="ru", api_key="test-key:fx")
        self.assertEqual(res, ["Привет", "Мир"])

    @patch("urllib.request.urlopen")
    def test_single_translation(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"translations": [{"text": "\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82"}]}'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        res = translate_single_with_deepl("Hello", target_lang="ru", api_key="test-key:fx")
        self.assertEqual(res, "Привет")

    @patch("urllib.request.urlopen")
    def test_cleaned_input_slot_and_http_cooldown(self, mock_urlopen):
        import urllib.error
        import deepl_translator

        with deepl_translator._deepl_cooldown_lock:
            deepl_translator._deepl_cooldown_until = 0.0

        # Empty slot for stripped garbage input must not drop valid sibling translations
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"translations": [{"text": ""}, {"text": "\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82"}]}'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        res = translate_dialogues_with_deepl(["8123' i ob'", "Hello"], target_lang="ru", api_key="test-key:fx")
        self.assertEqual(res, ["", "Привет"])

        # HTTP 429 must activate cooldown so immediate follow-up skips network call
        mock_urlopen.reset_mock()
        mock_urlopen.side_effect = urllib.error.HTTPError("https://api-free.deepl.com/v2/translate", 429, "Too Many", {}, None)
        self.assertIsNone(translate_dialogues_with_deepl(["Hello"], target_lang="ru", api_key="test-key:fx"))
        self.assertEqual(mock_urlopen.call_count, 1)

        self.assertIsNone(translate_dialogues_with_deepl(["Hello again"], target_lang="ru", api_key="test-key:fx"))
        self.assertEqual(mock_urlopen.call_count, 1)
        with deepl_translator._deepl_cooldown_lock:
            deepl_translator._deepl_cooldown_until = 0.0
