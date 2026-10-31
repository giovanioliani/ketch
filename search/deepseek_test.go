package search

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/1broseidon/ketch/health"
	config "github.com/1broseidon/ketch/internal/configbase"
)

func deepseekTestServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	return server
}

func TestDeepSeekSearchRequestAndMapping(t *testing.T) {
	t.Parallel()
	server := deepseekTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %q, want POST", r.Method)
		}
		if r.URL.Path != "/v1/tools/search" {
			t.Errorf("path = %q, want /v1/tools/search", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer ds-test" {
			t.Errorf("Authorization = %q, want Bearer ds-test", got)
		}
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode body: %v", err)
		}
		if req["query"] != "golang" {
			t.Errorf("query = %v, want golang", req["query"])
		}
		if req["count"] != float64(5) {
			t.Errorf("count = %v, want 5", req["count"])
		}
		if req["need_content"] != false {
			t.Errorf("need_content = %v, want false (search stays small)", req["need_content"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"results":[
			{"rerank_score":2.4,"title":"Go Docs","url":"https://go.dev/doc/","snippet":"The Go docs","summary":"A long summary","content":"","site_name":"go.dev","date":"2026-06-01T00:00:00+00:00","authority":4},
			{"rerank_score":1.1,"title":"Go Blog","url":"https://go.dev/blog/","snippet":"","summary":"Only a summary"}
		]}`)
	})

	backend := &DeepSeek{keys: newKeyPool([]string{"ds-test"}), client: rewrittenClient(server.URL)}
	results, err := backend.Search(context.Background(), "golang", 5)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("len = %d, want 2", len(results))
	}
	if results[0].Title != "Go Docs" || results[0].URL != "https://go.dev/doc/" || results[0].Description != "The Go docs" {
		t.Errorf("first result = %+v", results[0])
	}
	if results[0].Content != "" {
		t.Errorf("content = %q, want empty without need_content", results[0].Content)
	}
	if results[1].Description != "Only a summary" {
		t.Errorf("Description = %q, want the summary fallback", results[1].Description)
	}
}

func TestDeepSeekClampsCountAndSkipsEmptyURLs(t *testing.T) {
	t.Parallel()
	server := deepseekTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		var req deepseekSearchRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode: %v", err)
		}
		if req.Count != deepseekMaxResults {
			t.Errorf("count = %d, want clamp to %d", req.Count, deepseekMaxResults)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"results":[
			{"title":"no url"},
			{"title":"A","url":"https://a.example"},
			{"title":"B","url":"https://b.example"}
		]}`)
	})

	backend := &DeepSeek{keys: newKeyPool([]string{"ds-test"}), client: rewrittenClient(server.URL)}
	results, err := backend.Search(context.Background(), "q", 100)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("len(results) = %d, want 2 (empty URL skipped)", len(results))
	}
}

func TestDeepSeekZeroLimit(t *testing.T) {
	t.Parallel()
	results, err := NewDeepSeek("ds-test").Search(context.Background(), "q", 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 0 {
		t.Fatalf("got %d results, want 0", len(results))
	}
}

func TestDeepSeekStatusErrors(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name       string
		status     int
		body       string
		wantErrSub string
	}{
		{
			name:       "400 reads the API message",
			status:     http.StatusBadRequest,
			body:       `{"error":{"message":"'count' must be between 1 and 50","type":"invalid_request_error","param":null,"code":"invalid_request_error"}}`,
			wantErrSub: "'count' must be between 1 and 50",
		},
		{name: "401", status: http.StatusUnauthorized, body: `{"error":{"message":"bad key"}}`, wantErrSub: "deepseek_api_key"},
		{name: "402", status: http.StatusPaymentRequired, body: `{"error":{"message":"no balance"}}`, wantErrSub: "balance exhausted"},
		{name: "429", status: http.StatusTooManyRequests, body: `{"error":{"message":"slow down"}}`, wantErrSub: "rate limited"},
		{name: "500", status: http.StatusInternalServerError, body: `boom`, wantErrSub: "status 500"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			server := deepseekTestServer(t, func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tc.status)
				_, _ = io.WriteString(w, tc.body)
			})
			backend := &DeepSeek{keys: newKeyPool([]string{"ds-test"}), client: rewrittenClient(server.URL)}
			_, err := backend.Search(context.Background(), "q", 1)
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tc.wantErrSub) {
				t.Fatalf("error %q should contain %q", err.Error(), tc.wantErrSub)
			}
			if strings.Contains(err.Error(), "ds-test") {
				t.Fatalf("error leaked key: %s", err.Error())
			}
		})
	}
}

func TestDeepSeekMalformedJSON(t *testing.T) {
	t.Parallel()
	server := deepseekTestServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{not-json`)
	})
	backend := &DeepSeek{keys: newKeyPool([]string{"ds-test"}), client: rewrittenClient(server.URL)}
	_, err := backend.Search(context.Background(), "q", 1)
	if err == nil || !strings.Contains(err.Error(), "failed to decode deepseek search response") {
		t.Fatalf("error = %v, want decode failure", err)
	}
}

func TestDeepSeekRotatesKeyOnRetryableStatus(t *testing.T) {
	t.Parallel()
	for _, status := range []int{http.StatusUnauthorized, http.StatusTooManyRequests, http.StatusInternalServerError} {
		t.Run(http.StatusText(status), func(t *testing.T) {
			t.Parallel()
			var got []string
			server := deepseekTestServer(t, func(w http.ResponseWriter, r *http.Request) {
				got = append(got, strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer "))
				if len(got) == 1 {
					w.WriteHeader(status)
					return
				}
				_, _ = io.WriteString(w, `{"results":[{"title":"A","url":"https://a.example"}]}`)
			})
			backend := &DeepSeek{keys: deterministicPool("first-key", "second-key"), client: rewrittenClient(server.URL)}
			if _, err := backend.Search(context.Background(), "q", 1); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, []string{"first-key", "second-key"}) {
				t.Fatalf("keys used = %v, want first then a different key", got)
			}
		})
	}
}

func TestDeepSeekSingleKeyDoesNotRetry(t *testing.T) {
	t.Parallel()
	var calls int
	server := deepseekTestServer(t, func(w http.ResponseWriter, _ *http.Request) {
		calls++
		w.WriteHeader(http.StatusTooManyRequests)
	})
	backend := &DeepSeek{keys: newKeyPool([]string{"only-key"}), client: rewrittenClient(server.URL)}
	if _, err := backend.Search(context.Background(), "q", 1); err == nil {
		t.Fatal("expected a rate-limit error")
	}
	if calls != 1 {
		t.Fatalf("calls = %d, want 1 without another key to rotate to", calls)
	}
}

func TestDeepSeekFetchPage(t *testing.T) {
	t.Parallel()
	server := deepseekTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/tools/fetch" {
			t.Errorf("path = %q, want /v1/tools/fetch", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer ds-test" {
			t.Errorf("Authorization = %q, want Bearer ds-test", got)
		}
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode body: %v", err)
		}
		if req["url"] != "https://example.com/article" {
			t.Errorf("url = %v", req["url"])
		}
		_, _ = io.WriteString(w, `{"title":"Example article","content":"# Example article\n\nArticle content."}`)
	})

	backend := &DeepSeek{keys: newKeyPool([]string{"ds-test"}), client: rewrittenClient(server.URL)}
	title, markdown, err := backend.FetchPage(context.Background(), "https://example.com/article")
	if err != nil {
		t.Fatalf("FetchPage: %v", err)
	}
	if title != "Example article" {
		t.Errorf("title = %q", title)
	}
	if markdown != "# Example article\n\nArticle content." {
		t.Errorf("markdown = %q", markdown)
	}
}

func TestDeepSeekFetchErrors(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name       string
		status     int
		body       string
		wantErrSub string
	}{
		{name: "401", status: http.StatusUnauthorized, body: `{"error":{"message":"bad key"}}`, wantErrSub: "deepseek_api_key"},
		{name: "402", status: http.StatusPaymentRequired, body: `{"error":{"message":"no balance"}}`, wantErrSub: "balance exhausted"},
		{name: "403 policy", status: http.StatusForbidden, body: `{"error":{"message":"blocked by policy"}}`, wantErrSub: "blocked by policy"},
		{name: "404 unretrievable", status: http.StatusNotFound, body: `{"error":{"message":"no markdown"}}`, wantErrSub: "no usable page content"},
		{name: "429", status: http.StatusTooManyRequests, body: `{"error":{"message":"slow"}}`, wantErrSub: "rate limited"},
		{name: "504", status: http.StatusGatewayTimeout, body: ``, wantErrSub: "timed out"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			server := deepseekTestServer(t, func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tc.status)
				_, _ = io.WriteString(w, tc.body)
			})
			backend := &DeepSeek{keys: newKeyPool([]string{"ds-test"}), client: rewrittenClient(server.URL)}
			_, _, err := backend.FetchPage(context.Background(), "https://example.com")
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), tc.wantErrSub) {
				t.Fatalf("error %q should contain %q", err.Error(), tc.wantErrSub)
			}
		})
	}
}

func TestDeepSeekFetchEmptyContentIsAnError(t *testing.T) {
	t.Parallel()
	server := deepseekTestServer(t, func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"title":"Empty","content":"   "}`)
	})
	backend := &DeepSeek{keys: newKeyPool([]string{"ds-test"}), client: rewrittenClient(server.URL)}
	_, _, err := backend.FetchPage(context.Background(), "https://example.com")
	if err == nil || !strings.Contains(err.Error(), "no content") {
		t.Fatalf("error = %v, want empty-content failure", err)
	}
}

func TestProbeDeepSeek(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name       string
		status     int
		key        string
		wantStatus health.Status
	}{
		{name: "no key", key: "", wantStatus: health.StatusNoKey},
		{name: "ok", status: http.StatusOK, key: "ds-test", wantStatus: health.StatusOK},
		{name: "rejected", status: http.StatusUnauthorized, key: "ds-test", wantStatus: health.StatusMisconfigured},
		{name: "no balance", status: http.StatusPaymentRequired, key: "ds-test", wantStatus: health.StatusOK},
		{name: "rate limited", status: http.StatusTooManyRequests, key: "ds-test", wantStatus: health.StatusOK},
		{name: "server error", status: http.StatusInternalServerError, key: "ds-test", wantStatus: health.StatusUnreachable},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			server := deepseekTestServer(t, func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v1/tools/search" {
					t.Errorf("probe path = %q", r.URL.Path)
				}
				if tc.key != "" && r.Header.Get("Authorization") != "Bearer "+tc.key {
					t.Errorf("Authorization = %q", r.Header.Get("Authorization"))
				}
				w.WriteHeader(tc.status)
			})
			client := rewrittenClient(server.URL)
			got, detail := ProbeDeepSeek(context.Background(), client, tc.key)
			if got != tc.wantStatus {
				t.Fatalf("status = %q (%s), want %q", got, detail, tc.wantStatus)
			}
		})
	}
}

func TestDeepSeekProviderDescriptor(t *testing.T) {
	t.Parallel()
	p, ok := Lookup("deepseek")
	if !ok {
		t.Fatal("deepseek is not registered")
	}
	if p.Name != "DeepSeek" || p.AutoRank == 0 || p.Setup == "" || p.Probe == nil || p.Usable == nil || p.New == nil {
		t.Fatalf("incomplete descriptor: %+v", p)
	}
	if !p.Fetches {
		t.Error("deepseek must be selectable as a fetch backend")
	}
	if p.Usable(&config.Config{}) {
		t.Error("deepseek must be unusable without a key")
	}
	cfg := config.Config{ProviderSettings: map[string]any{"deepseek_api_key": "ds-test"}}
	if !p.Usable(&cfg) {
		t.Error("deepseek must be usable with a key")
	}
	if _, err := p.New(&cfg); err != nil {
		t.Fatalf("New: %v", err)
	}
}

func TestFetchBackendsListing(t *testing.T) {
	t.Parallel()
	backends := FetchBackends()
	if backends[0] != FetchBackendHTTP {
		t.Fatalf("FetchBackends()[0] = %q, want %q", backends[0], FetchBackendHTTP)
	}
	if !slices.Contains(backends, "deepseek") {
		t.Fatalf("FetchBackends() = %v, missing deepseek", backends)
	}
	for _, id := range backends {
		if !IsFetchBackend(id) {
			t.Errorf("IsFetchBackend(%q) = false", id)
		}
	}
}

func TestIsFetchBackend(t *testing.T) {
	t.Parallel()
	for _, id := range []string{"", "HTTP", "deepseek", "http"} {
		if !IsFetchBackend(id) {
			t.Errorf("IsFetchBackend(%q) = false", id)
		}
	}
	for _, id := range []string{"tavily", "TAVILY", "nope"} {
		if IsFetchBackend(id) {
			t.Errorf("IsFetchBackend(%q) = true, but it is not a fetch backend", id)
		}
	}
}

func TestNewFetcherFromConfig(t *testing.T) {
	t.Parallel()

	t.Run("deepseek with a key", func(t *testing.T) {
		t.Parallel()
		cfg := config.Config{FetchBackend: "deepseek", ProviderSettings: map[string]any{"deepseek_api_key": "ds-test"}}
		fetcher, err := NewFetcherFromConfig(&cfg)
		if err != nil {
			t.Fatalf("NewFetcherFromConfig: %v", err)
		}
		if _, ok := fetcher.(*DeepSeek); !ok {
			t.Fatalf("fetcher = %T, want *DeepSeek", fetcher)
		}
	})

	t.Run("default and explicit http return no fetcher", func(t *testing.T) {
		t.Parallel()
		for _, cfg := range []*config.Config{{}, {FetchBackend: FetchBackendHTTP}} {
			fetcher, err := NewFetcherFromConfig(cfg)
			if err != nil || fetcher != nil {
				t.Fatalf("fetcher = %v, err = %v; want nil, nil", fetcher, err)
			}
		}
	})

	t.Run("missing key", func(t *testing.T) {
		t.Parallel()
		_, err := NewFetcherFromConfig(&config.Config{FetchBackend: "deepseek"})
		if err == nil || !strings.Contains(err.Error(), "deepseek_api_key") {
			t.Fatalf("missing key error = %v", err)
		}
	})

	t.Run("unknown backend", func(t *testing.T) {
		t.Parallel()
		_, err := NewFetcherFromConfig(&config.Config{FetchBackend: "nope"})
		if err == nil || !strings.Contains(err.Error(), "unknown fetch backend") {
			t.Fatalf("unknown backend error = %v", err)
		}
	})
}

func TestDeepSeekFetchBackendGatesDoctor(t *testing.T) {
	t.Parallel()
	p, ok := Lookup("deepseek")
	if !ok {
		t.Fatal("deepseek is not registered")
	}
	cfg := config.Config{FetchBackend: "deepseek"}
	if !p.Required(&cfg) {
		t.Error("a selected fetch_backend must be required, so doctor fails without its key")
	}
	cfg.FetchBackend = FetchBackendHTTP
	if p.Required(&cfg) {
		t.Error("an unselected, unconfigured provider must not be required")
	}
	cfg.FetchBackend = ""
	if p.Required(&cfg) {
		t.Error("the default fetch backend is http, not deepseek")
	}
}

func TestDeepSeekRequestErrorDoesNotLeakKey(t *testing.T) {
	t.Parallel()
	client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		return nil, errors.New("dial tcp: connection refused")
	})}
	backend := &DeepSeek{keys: newKeyPool([]string{"super-secret"}), client: client}
	_, err := backend.Search(context.Background(), "q", 1)
	if err == nil {
		t.Fatal("expected transport error")
	}
	if strings.Contains(err.Error(), "super-secret") {
		t.Fatalf("error leaked key: %v", err)
	}
}
