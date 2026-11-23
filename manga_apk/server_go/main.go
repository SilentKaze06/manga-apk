package main

import (
	"context"
	"crypto/subtle"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"html"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"
)

// Configuration from environment variables
var (
	port                  = getEnv("PORT", "8000")
	bindHost              = getEnv("MANGA_BIND_HOST", "0.0.0.0")
	serverAPIKey          = strings.TrimSpace(os.Getenv("MANGA_SERVER_API_KEY"))
	allowUnauthenticated  = strings.ToLower(strings.TrimSpace(getEnv("MANGA_ALLOW_UNAUTHENTICATED", "false"))) == "true"
	translationTimeoutSec = 5 * time.Second
	maxUploadBytes  int64 = 15 * 1024 * 1024 // 15 MB
)

func getEnv(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}

var defaultTransport = http.DefaultTransport.(*http.Transport).Clone()

func init() {
	if defaultTransport.TLSClientConfig == nil {
		defaultTransport.TLSClientConfig = &tls.Config{}
	}
	defaultTransport.TLSClientConfig.Renegotiation = tls.RenegotiateOnceAsClient
	defaultTransport.MaxIdleConns = 100
	defaultTransport.MaxIdleConnsPerHost = 20
	defaultTransport.IdleConnTimeout = 90 * time.Second
}

var httpClient = &http.Client{
	Timeout:   translationTimeoutSec,
	Transport: defaultTransport,
}

// Language code normalizer
func normalizeLang(code string) string {
	code = strings.ToLower(strings.TrimSpace(code))
	switch code {
	case "jpn", "japanese":
		return "ja"
	case "kor", "korean":
		return "ko"
	case "chs", "cht", "chinese", "ch":
		return "zh"
	case "eng", "english":
		return "en"
	case "rus", "russian":
		return "ru"
	case "uzb", "uzbek":
		return "uz"
	case "":
		return "auto"
	default:
		return code
	}
}

func translationLangCode(code string) string {
	code = normalizeLang(code)
	if code == "zh" {
		return "zh-CN"
	}
	return code
}

// ─── Fast Google Translate Client ─────────────────────────────────────────────

func translateSingleText(ctx context.Context, text, targetLang, sourceLang string) (string, error) {
	cleanText := strings.TrimSpace(text)
	if cleanText == "" {
		return "", nil
	}

	targetClean := strings.TrimSpace(targetLang)
	if targetClean == "" {
		targetClean = "uz"
	}
	srcParam := translationLangCode(sourceLang)
	targetParam := translationLangCode(targetClean)
	if targetParam == "" {
		targetParam = "uz"
	}

	// 1. Google Chrome Extension Translate API (Super fast ~50ms, returns JSON ["translated text"])
	chromeURL := fmt.Sprintf(
		"https://clients5.google.com/translate_a/t?client=dict-chrome-ex&sl=%s&tl=%s&q=%s",
		url.QueryEscape(srcParam),
		url.QueryEscape(targetParam),
		url.QueryEscape(cleanText),
	)

	req1, err := http.NewRequestWithContext(ctx, http.MethodGet, chromeURL, nil)
	if err == nil {
		resp, err := httpClient.Do(req1)
		if err == nil {
			defer resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				body, _ := io.ReadAll(io.LimitReader(resp.Body, 128*1024))
				var translations []string
				if err := json.Unmarshal(body, &translations); err == nil && len(translations) > 0 {
					result := strings.TrimSpace(translations[0])
					if result != "" {
						return result, nil
					}
				}
			}
		}
	}

	// 2. Google Translate GTX API (Fallback)
	gtxURL := fmt.Sprintf(
		"https://translate.googleapis.com/translate_a/single?client=gtx&dt=t&sl=%s&tl=%s&q=%s",
		url.QueryEscape(srcParam),
		url.QueryEscape(targetParam),
		url.QueryEscape(cleanText),
	)

	req2, err := http.NewRequestWithContext(ctx, http.MethodGet, gtxURL, nil)
	if err == nil {
		req2.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
		resp2, err := httpClient.Do(req2)
		if err == nil {
			defer resp2.Body.Close()
			if resp2.StatusCode == http.StatusOK {
				body, _ := io.ReadAll(io.LimitReader(resp2.Body, 512*1024))
				var rawData []interface{}
				if err := json.Unmarshal(body, &rawData); err == nil && len(rawData) > 0 {
					if segments, ok := rawData[0].([]interface{}); ok {
						var sb strings.Builder
						for _, seg := range segments {
							if parts, ok := seg.([]interface{}); ok && len(parts) > 0 {
								if translatedPart, ok := parts[0].(string); ok {
									sb.WriteString(translatedPart)
								}
							}
						}
						result := strings.TrimSpace(sb.String())
						if result != "" {
							return result, nil
						}
					}
				}
			}
		}
	}

	// 3. Google Translate Web Fallback
	webURL := fmt.Sprintf(
		"https://translate.google.com/m?sl=%s&tl=%s&q=%s",
		url.QueryEscape(srcParam),
		url.QueryEscape(targetParam),
		url.QueryEscape(cleanText),
	)
	req3, err := http.NewRequestWithContext(ctx, http.MethodGet, webURL, nil)
	if err == nil {
		req3.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
		resp3, err := httpClient.Do(req3)
		if err == nil {
			defer resp3.Body.Close()
			if resp3.StatusCode == http.StatusOK {
				body, _ := io.ReadAll(io.LimitReader(resp3.Body, 512*1024))
				htmlStr := string(body)
				re := regexp.MustCompile(`class="(?:result-container|t0)">([^<]+)<`)
				matches := re.FindStringSubmatch(htmlStr)
				if len(matches) > 1 {
					translated := html.UnescapeString(strings.TrimSpace(matches[1]))
					if translated != "" {
						return translated, nil
					}
				}
			}
		}
	}

	return cleanText, fmt.Errorf("tarjima xizmatlaridan javob olinmadi")
}

// Batch translate multiple texts using bounded worker pool and context
func translateBatch(ctx context.Context, texts []string, targetLang, sourceLang string) []string {
	if len(texts) == 0 {
		return []string{}
	}

	results := make([]string, len(texts))
	type job struct {
		index int
		text  string
	}

	// Limit concurrency to 8 parallel goroutines to avoid rate limits
	numWorkers := 8
	if len(texts) < numWorkers {
		numWorkers = len(texts)
	}

	jobs := make(chan job, len(texts))
	for i, t := range texts {
		jobs <- job{index: i, text: t}
	}
	close(jobs)

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-ctx.Done():
					return
				case j, ok := <-jobs:
					if !ok {
						return
					}
					translated, err := translateSingleText(ctx, j.text, targetLang, sourceLang)
					if err != nil || translated == "" {
						results[j.index] = j.text
					} else {
						results[j.index] = translated
					}
				}
			}
		}()
	}

	wg.Wait()

	// Fill any empty results with original text (e.g. if ctx was canceled or failed)
	for i, r := range results {
		if r == "" && texts[i] != "" {
			results[i] = texts[i]
		}
	}

	return results
}

// ─── Authentication & Helpers ────────────────────────────────────────────────

func sendJSONError(w http.ResponseWriter, message string, statusCode int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"detail": message,
	})
}

func checkAuth(r *http.Request) bool {
	if serverAPIKey == "" && allowUnauthenticated {
		return true
	}
	if serverAPIKey == "" {
		return false
	}
	clientKey := strings.TrimSpace(r.Header.Get("X-API-Key"))
	if clientKey == "" {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(clientKey), []byte(serverAPIKey)) == 1
}

// ─── CORS Middleware ─────────────────────────────────────────────────────────

func corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, X-API-Key, Authorization, Accept, Origin, User-Agent")
		w.Header().Set("Access-Control-Max-Age", "86400")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// ─── HTTP Handler Functions ───────────────────────────────────────────────────

func handleHealth(w http.ResponseWriter, r *http.Request) {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"status":         "ok",
		"server":         "manga-server-go",
		"version":        "2.0.0-high-performance",
		"arch":           runtime.GOARCH,
		"os":             runtime.GOOS,
		"goroutines":     runtime.NumGoroutine(),
		"alloc_ram_mb":   fmt.Sprintf("%.2f MB", float64(m.Alloc)/(1024*1024)),
		"total_alloc_mb": fmt.Sprintf("%.2f MB", float64(m.TotalAlloc)/(1024*1024)),
		"sys_ram_mb":     fmt.Sprintf("%.2f MB", float64(m.Sys)/(1024*1024)),
		"num_gc":         m.NumGC,
		"auth_required":  serverAPIKey != "" || !allowUnauthenticated,
	})
}

func handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"title":       "Ultra-Fast Manga & Manhwa OCR Translation Server (Go)",
		"description": "High-concurrency, low-memory translation engine for Android Manga App",
		"version":     "2.0.0",
		"endpoints": []string{
			"GET  /health",
			"POST /translate-manga",
			"POST /ocr-manga",
			"POST /detect-and-translate",
			"POST /translate-batch",
		},
	})
}

// Single bubble translation handler (/translate-manga, /ocr-manga)
func handleTranslateManga(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendJSONError(w, "Faqat POST so'rovi qabul qilinadi", http.StatusMethodNotAllowed)
		return
	}
	if !checkAuth(r) {
		if serverAPIKey == "" && !allowUnauthenticated {
			sendJSONError(w, "Server API kaliti sozlanmagan. Mahalliy sinov uchun MANGA_ALLOW_UNAUTHENTICATED=true ni belgilang.", http.StatusServiceUnavailable)
		} else {
			sendJSONError(w, "Server API kaliti noto'g'ri yoki kiritilmagan.", http.StatusUnauthorized)
		}
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, maxUploadBytes)
	contentType := r.Header.Get("Content-Type")
	if strings.Contains(contentType, "multipart/form-data") {
		if err := r.ParseMultipartForm(maxUploadBytes); err != nil {
			sendJSONError(w, fmt.Sprintf("Yuklangan ma'lumot hajmi juda katta yoki xato: %v", err), http.StatusBadRequest)
			return
		}
		if r.MultipartForm != nil {
			defer r.MultipartForm.RemoveAll()
		}
	} else {
		if err := r.ParseForm(); err != nil {
			sendJSONError(w, fmt.Sprintf("Form ma'lumoti xato: %v", err), http.StatusBadRequest)
			return
		}
	}

	rawTargetLang := strings.TrimSpace(r.FormValue("target_lang"))
	if rawTargetLang == "" {
		rawTargetLang = "uz"
	}
	targetLang := normalizeLang(rawTargetLang)
	lang := normalizeLang(r.FormValue("lang"))

	// Check if text was passed directly (fast path)
	directText := strings.TrimSpace(r.FormValue("text"))

	// If image-only OCR or /ocr-manga, return 501 until ONNX runtime is wired
	if r.URL.Path == "/ocr-manga" || directText == "" {
		sendJSONError(w, "Image-only OCR is not implemented yet in Go server (ONNX runtime not wired)", http.StatusNotImplemented)
		return
	}

	translatedText := ""
	if directText != "" {
		translatedText, _ = translateSingleText(r.Context(), directText, targetLang, lang)
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"status":          "success",
		"original_text":   directText,
		"translated_text": translatedText,
		"lang":            lang,
		"target_lang":     targetLang,
	})
}

// Whole-screen detection and translation handler (/detect-and-translate)
func handleDetectAndTranslate(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendJSONError(w, "Faqat POST so'rovi qabul qilinadi", http.StatusMethodNotAllowed)
		return
	}
	if !checkAuth(r) {
		if serverAPIKey == "" && !allowUnauthenticated {
			sendJSONError(w, "Server API kaliti sozlanmagan. Mahalliy sinov uchun MANGA_ALLOW_UNAUTHENTICATED=true ni belgilang.", http.StatusServiceUnavailable)
		} else {
			sendJSONError(w, "Server API kaliti noto'g'ri yoki kiritilmagan.", http.StatusUnauthorized)
		}
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, maxUploadBytes)
	contentType := r.Header.Get("Content-Type")
	if strings.Contains(contentType, "multipart/form-data") {
		if err := r.ParseMultipartForm(maxUploadBytes); err != nil {
			sendJSONError(w, fmt.Sprintf("Yuklangan ma'lumot hajmi juda katta yoki xato: %v", err), http.StatusBadRequest)
			return
		}
		if r.MultipartForm != nil {
			defer r.MultipartForm.RemoveAll()
		}
	} else {
		if err := r.ParseForm(); err != nil {
			sendJSONError(w, fmt.Sprintf("Form ma'lumoti xato: %v", err), http.StatusBadRequest)
			return
		}
	}

	// Whole-screen detection requires ONNX runtime which is not wired yet
	sendJSONError(w, "Whole-screen bubble detection and translation is not implemented yet in Go server (ONNX runtime not wired)", http.StatusNotImplemented)
}

// High-speed batch translation endpoint (/translate-batch)
func handleTranslateBatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		sendJSONError(w, "Faqat POST so'rovi qabul qilinadi", http.StatusMethodNotAllowed)
		return
	}
	if !checkAuth(r) {
		if serverAPIKey == "" && !allowUnauthenticated {
			sendJSONError(w, "Server API kaliti sozlanmagan. Mahalliy sinov uchun MANGA_ALLOW_UNAUTHENTICATED=true ni belgilang.", http.StatusServiceUnavailable)
		} else {
			sendJSONError(w, "Server API kaliti noto'g'ri yoki kiritilmagan.", http.StatusUnauthorized)
		}
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, maxUploadBytes)
	var reqBody struct {
		Texts      []string `json:"texts"`
		SourceLang string   `json:"source_lang"`
		TargetLang string   `json:"target_lang"`
	}

	if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
		sendJSONError(w, "JSON formati noto'g'ri", http.StatusBadRequest)
		return
	}

	rawTarget := strings.TrimSpace(reqBody.TargetLang)
	if rawTarget == "" {
		rawTarget = "uz"
	}
	targetLang := normalizeLang(rawTarget)
	sourceLang := normalizeLang(reqBody.SourceLang)

	startTime := time.Now()
	translated := translateBatch(r.Context(), reqBody.Texts, targetLang, sourceLang)
	if translated == nil {
		translated = []string{}
	}
	elapsed := time.Since(startTime)

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"status":       "success",
		"source_lang":  sourceLang,
		"target_lang":  targetLang,
		"count":        len(translated),
		"translations": translated,
		"elapsed_ms":   elapsed.Milliseconds(),
	})
}

// ─── Main Server Loop ─────────────────────────────────────────────────────────

func main() {
	mux := http.NewServeMux()

	mux.HandleFunc("/", handleRoot)
	mux.HandleFunc("/health", handleHealth)
	mux.HandleFunc("/translate-manga", handleTranslateManga)
	mux.HandleFunc("/ocr-manga", handleTranslateManga)
	mux.HandleFunc("/detect-and-translate", handleDetectAndTranslate)
	mux.HandleFunc("/translate-batch", handleTranslateBatch)

	addr := fmt.Sprintf("%s:%s", bindHost, port)
	log.Println("============================================================")
	log.Printf("  Manga & Manhwa Ultra-Fast Go Server (v2.0.0)")
	log.Printf("  Tinglash manzili: http://%s", addr)
	log.Printf("  Autentifikatsiya: API Key = %t, Allow Unauth = %t", serverAPIKey != "", allowUnauthenticated)
	log.Printf("  Operativ xotira (RAM) sarfi: ~15-25 MB (PyTorch 2.5GB o'rniga)")
	log.Println("============================================================")

	server := &http.Server{
		Addr:         addr,
		Handler:      corsMiddleware(mux),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	serverErrors := make(chan error, 1)
	go func() {
		log.Printf("Server ishga tushdi: http://%s", addr)
		serverErrors <- server.ListenAndServe()
	}()

	shutdown := make(chan os.Signal, 1)
	signal.Notify(shutdown, os.Interrupt, syscall.SIGTERM)

	select {
	case err := <-serverErrors:
		if err != nil && err != http.ErrServerClosed {
			log.Fatalf("Server xatosi: %v", err)
		}
	case sig := <-shutdown:
		log.Printf("Serverni to'xtatish signali qabul qilindi: %v. To'xtatilmoqda...", sig)
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()

		if err := server.Shutdown(ctx); err != nil {
			log.Printf("Server majburiy to'xtatildi: %v", err)
			_ = server.Close()
		}
		log.Println("Server to'liq to'xtatildi.")
	}
}
