package main

import (
	"bytes"
	"context"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHandleRoot(t *testing.T) {
	// Valid root path
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	rec := httptest.NewRecorder()
	handleRoot(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rec.Code)
	}
	if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("expected application/json, got %s", ct)
	}

	// Non-root path should return 404
	req404 := httptest.NewRequest(http.MethodGet, "/random-not-found", nil)
	rec404 := httptest.NewRecorder()
	handleRoot(rec404, req404)

	if rec404.Code != http.StatusNotFound {
		t.Fatalf("expected 404, got %d", rec404.Code)
	}
}

func TestCorsMiddleware(t *testing.T) {
	dummy := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	handler := corsMiddleware(dummy)

	// OPTIONS preflight
	reqOpt := httptest.NewRequest(http.MethodOptions, "/translate-manga", nil)
	recOpt := httptest.NewRecorder()
	handler.ServeHTTP(recOpt, reqOpt)

	if recOpt.Code != http.StatusNoContent {
		t.Errorf("expected 204 for OPTIONS, got %d", recOpt.Code)
	}
	if allow := recOpt.Header().Get("Access-Control-Allow-Origin"); allow != "*" {
		t.Errorf("expected Access-Control-Allow-Origin: *, got %s", allow)
	}

	// Regular GET
	reqGet := httptest.NewRequest(http.MethodGet, "/health", nil)
	recGet := httptest.NewRecorder()
	handler.ServeHTTP(recGet, reqGet)

	if recGet.Header().Get("Access-Control-Allow-Origin") != "*" {
		t.Errorf("expected CORS header on GET")
	}
}

func TestCheckAuth(t *testing.T) {
	origKey := serverAPIKey
	origAllow := allowUnauthenticated
	defer func() {
		serverAPIKey = origKey
		allowUnauthenticated = origAllow
	}()

	// Case 1: unauthenticated disallowed, no key configured
	serverAPIKey = ""
	allowUnauthenticated = false
	req := httptest.NewRequest(http.MethodPost, "/translate-manga", nil)
	if checkAuth(req) {
		t.Errorf("expected checkAuth to fail when unauthenticated disallowed")
	}

	// Case 2: unauthenticated allowed
	serverAPIKey = ""
	allowUnauthenticated = true
	if !checkAuth(req) {
		t.Errorf("expected checkAuth to pass when allowUnauthenticated=true")
	}

	// Case 3: server key configured, matching client key
	serverAPIKey = "secret-token-123"
	allowUnauthenticated = false
	reqWithKey := httptest.NewRequest(http.MethodPost, "/translate-manga", nil)
	reqWithKey.Header.Set("X-API-Key", "secret-token-123")
	if !checkAuth(reqWithKey) {
		t.Errorf("expected checkAuth to pass with valid X-API-Key")
	}

	// Case 4: server key configured, mismatch
	reqWrongKey := httptest.NewRequest(http.MethodPost, "/translate-manga", nil)
	reqWrongKey.Header.Set("X-API-Key", "wrong-token")
	if checkAuth(reqWrongKey) {
		t.Errorf("expected checkAuth to fail with wrong X-API-Key")
	}
}

func TestLanguageNormalizationAndDefault(t *testing.T) {
	// Raw target lang default
	rawTarget := ""
	if rawTarget == "" {
		rawTarget = "uz"
	}
	targetLang := normalizeLang(rawTarget)
	if targetLang != "uz" {
		t.Fatalf("expected uz, got %s", targetLang)
	}

	if normalizeLang("jpn") != "ja" {
		t.Errorf("expected ja for jpn")
	}
	if normalizeLang("kor") != "ko" {
		t.Errorf("expected ko for kor")
	}
	if normalizeLang("chs") != "zh" {
		t.Errorf("expected zh for chs")
	}
	if translationLangCode("zh") != "zh-CN" {
		t.Errorf("expected zh-CN for translationLangCode(zh)")
	}
}

func TestImageOnly501(t *testing.T) {
	origAllow := allowUnauthenticated
	allowUnauthenticated = true
	defer func() { allowUnauthenticated = origAllow }()

	// /ocr-manga should return 501
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	part, _ := writer.CreateFormFile("file", "test.png")
	part.Write([]byte("fake-png-data"))
	writer.Close()

	reqOcr := httptest.NewRequest(http.MethodPost, "/ocr-manga", body)
	reqOcr.Header.Set("Content-Type", writer.FormDataContentType())
	recOcr := httptest.NewRecorder()
	handleTranslateManga(recOcr, reqOcr)

	if recOcr.Code != http.StatusNotImplemented {
		t.Fatalf("expected 501 for /ocr-manga, got %d", recOcr.Code)
	}
	if ct := recOcr.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("expected application/json error, got %s", ct)
	}

	// /detect-and-translate should return 501
	bodyDetect := &bytes.Buffer{}
	writerDetect := multipart.NewWriter(bodyDetect)
	partDetect, _ := writerDetect.CreateFormFile("file", "test.png")
	partDetect.Write([]byte("fake-png-data"))
	writerDetect.Close()

	reqDetect := httptest.NewRequest(http.MethodPost, "/detect-and-translate", bodyDetect)
	reqDetect.Header.Set("Content-Type", writerDetect.FormDataContentType())
	recDetect := httptest.NewRecorder()
	handleDetectAndTranslate(recDetect, reqDetect)

	if recDetect.Code != http.StatusNotImplemented {
		t.Fatalf("expected 501 for /detect-and-translate, got %d", recDetect.Code)
	}
	if ct := recDetect.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("expected application/json error, got %s", ct)
	}

	// /translate-manga with image only and no text should return 501
	bodyImgOnly := &bytes.Buffer{}
	writerImgOnly := multipart.NewWriter(bodyImgOnly)
	partImg, _ := writerImgOnly.CreateFormFile("file", "test.png")
	partImg.Write([]byte("fake-png-data"))
	writerImgOnly.Close()

	reqImgOnly := httptest.NewRequest(http.MethodPost, "/translate-manga", bodyImgOnly)
	reqImgOnly.Header.Set("Content-Type", writerImgOnly.FormDataContentType())
	recImgOnly := httptest.NewRecorder()
	handleTranslateManga(recImgOnly, reqImgOnly)

	if recImgOnly.Code != http.StatusNotImplemented {
		t.Fatalf("expected 501 for image-only /translate-manga, got %d", recImgOnly.Code)
	}
}

func TestTranslateBatch(t *testing.T) {
	origAllow := allowUnauthenticated
	allowUnauthenticated = true
	defer func() { allowUnauthenticated = origAllow }()

	// Empty list should return empty slice, not nil/null
	res := translateBatch(context.Background(), []string{}, "uz", "en")
	if res == nil || len(res) != 0 {
		t.Fatalf("expected non-nil empty slice, got %#v", res)
	}

	// JSON endpoint with empty list
	reqEmpty := httptest.NewRequest(http.MethodPost, "/translate-batch", strings.NewReader(`{"texts":[],"target_lang":"uz"}`))
	reqEmpty.Header.Set("Content-Type", "application/json")
	recEmpty := httptest.NewRecorder()
	handleTranslateBatch(recEmpty, reqEmpty)

	if recEmpty.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", recEmpty.Code)
	}
	var respData struct {
		Status       string   `json:"status"`
		Translations []string `json:"translations"`
	}
	if err := json.Unmarshal(recEmpty.Body.Bytes(), &respData); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if respData.Translations == nil {
		t.Errorf("expected non-nil translations array in JSON response")
	}

	// Bad JSON should return 400 with application/json
	reqBad := httptest.NewRequest(http.MethodPost, "/translate-batch", strings.NewReader(`{invalid-json`))
	reqBad.Header.Set("Content-Type", "application/json")
	recBad := httptest.NewRecorder()
	handleTranslateBatch(recBad, reqBad)

	if recBad.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", recBad.Code)
	}
	if ct := recBad.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("expected application/json error, got %s", ct)
	}
}

func TestMaxBytesReaderEnforcement(t *testing.T) {
	origAllow := allowUnauthenticated
	origMax := maxUploadBytes
	allowUnauthenticated = true
	maxUploadBytes = 100 // Set small limit for testing
	defer func() {
		allowUnauthenticated = origAllow
		maxUploadBytes = origMax
	}()

	largePayload := strings.Repeat("A", 1000)
	req := httptest.NewRequest(http.MethodPost, "/translate-batch", strings.NewReader(largePayload))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	handleTranslateBatch(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for payload exceeding maxUploadBytes, got %d", rec.Code)
	}
}
