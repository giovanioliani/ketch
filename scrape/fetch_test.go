package scrape

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/1broseidon/ketch/config"
	"github.com/1broseidon/ketch/search"
)

// stubPageFetcher stands in for a provider fetch backend (DeepSeek Fetch).
type stubPageFetcher struct {
	gotURL   string
	title    string
	markdown string
	err      error
	calls    int
}

func (f *stubPageFetcher) FetchPage(_ context.Context, rawURL string) (string, string, error) {
	f.calls++
	f.gotURL = rawURL
	return f.title, f.markdown, f.err
}

func scraperWithFetcher(t *testing.T, f search.PageFetcher) *Scraper {
	t.Helper()
	s := NewWithRewriter("", nil)
	s.fetcher = f
	s.fetchBackend = "deepseek"
	t.Cleanup(s.Close)
	return s
}

func TestScrapeUsesFetchBackend(t *testing.T) {
	t.Parallel()
	f := &stubPageFetcher{title: "Example", markdown: "# Example\n\nBody."}
	s := scraperWithFetcher(t, f)

	page, source, err := s.Scrape(context.Background(), "https://example.com/article")
	if err != nil {
		t.Fatalf("Scrape: %v", err)
	}
	if f.calls != 1 || f.gotURL != "https://example.com/article" {
		t.Fatalf("fetcher calls=%d url=%q", f.calls, f.gotURL)
	}
	if page.Title != "Example" || page.Markdown != "# Example\n\nBody." {
		t.Fatalf("page = %+v", page)
	}
	if page.ContentHash == "" {
		t.Error("ContentHash must be set so the page cache can detect changes")
	}
	if source != SourceFetch {
		t.Fatalf("source = %q, want %q", source, SourceFetch)
	}
}

func TestScrapeFetchBackendAppliesURLRewrites(t *testing.T) {
	t.Parallel()
	f := &stubPageFetcher{title: "Example", markdown: "Body."}
	s := scraperWithFetcher(t, f)

	page, _, err := s.Scrape(context.Background(), "https://old.example/article")
	if err != nil {
		t.Fatal(err)
	}
	// No rewriter configured, so the fetch URL equals the original and no
	// FetchedURL is recorded.
	if page.FetchedURL != "" {
		t.Fatalf("FetchedURL = %q, want empty", page.FetchedURL)
	}
}

func TestScrapeFetchBackendValidatesURL(t *testing.T) {
	t.Parallel()
	f := &stubPageFetcher{markdown: "Body."}
	s := scraperWithFetcher(t, f)

	_, _, err := s.Scrape(context.Background(), "file:///etc/passwd")
	if !errors.Is(err, ErrInvalidURL) {
		t.Fatalf("error = %v, want ErrInvalidURL", err)
	}
	if f.calls != 0 {
		t.Fatal("fetcher must not be called for a non-web URL")
	}
}

func TestScrapeFetchBackendErrorIsReturned(t *testing.T) {
	t.Parallel()
	f := &stubPageFetcher{err: errors.New("deepseek fetch: no usable page content")}
	s := scraperWithFetcher(t, f)

	_, _, err := s.Scrape(context.Background(), "https://example.com")
	if err == nil || !strings.Contains(err.Error(), "no usable page content") {
		t.Fatalf("error = %v", err)
	}
}

func TestCachedScrapeFetchBackendCachesByBackend(t *testing.T) {
	t.Parallel()
	f := &stubPageFetcher{title: "Example", markdown: "Body."}
	s := scraperWithFetcher(t, f)
	pc := newMapPageCache()
	url := "https://example.com/article"

	first, err := s.CachedScrape(context.Background(), pc, url)
	if err != nil {
		t.Fatal(err)
	}
	second, err := s.CachedScrape(context.Background(), pc, url)
	if err != nil {
		t.Fatal(err)
	}
	if f.calls != 1 {
		t.Fatalf("fetcher calls = %d, want 1 (second scrape is a cache hit)", f.calls)
	}
	if first.Markdown != second.Markdown {
		t.Fatalf("cached page = %+v, first = %+v", second, first)
	}

	plain := NewWithRewriter("", nil) // built-in HTTP pipeline
	if plain.CacheKey(url) == s.MarkdownCacheKey(url) {
		t.Fatal("fetch-backend pages must not share cache keys with the HTTP pipeline")
	}
}

// The raw and conditional paths need the page body, which a markdown fetch
// backend cannot provide. They must keep using HTTP even when one is
// configured.
func TestScrapeConditionalIgnoresFetchBackend(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = io.WriteString(w, `<html><head><title>HTTP page</title></head><body><p>Fetched over HTTP.</p></body></html>`)
	}))
	t.Cleanup(server.Close)

	f := &stubPageFetcher{markdown: "from the fetch backend"}
	s := scraperWithFetcher(t, f)

	result, err := s.ScrapeConditional(context.Background(), server.URL, "", "")
	if err != nil {
		t.Fatalf("ScrapeConditional: %v", err)
	}
	if f.calls != 0 {
		t.Fatal("raw/conditional scraping must not use a markdown fetch backend")
	}
	if result.Source != SourceHTTP || !strings.Contains(result.RawHTML, "Fetched over HTTP.") {
		t.Fatalf("result = %+v", result)
	}
	if result.Page == nil || result.Page.Title != "HTTP page" {
		t.Fatalf("page = %+v", result.Page)
	}
}

func TestNewFromConfigFetchBackend(t *testing.T) {
	t.Parallel()

	t.Run("default is the HTTP pipeline", func(t *testing.T) {
		t.Parallel()
		s := scraperFromConfig(t, &config.Config{})
		if s.fetcher != nil || s.fetchBackend != "" {
			t.Fatalf("fetcher = %v, backend = %q; want none", s.fetcher, s.fetchBackend)
		}
	})

	t.Run("deepseek needs a key", func(t *testing.T) {
		t.Parallel()
		_, err := NewFromConfig(&config.Config{FetchBackend: "deepseek"})
		if err == nil || !strings.Contains(err.Error(), "deepseek_api_key") {
			t.Fatalf("error = %v", err)
		}
	})

	t.Run("deepseek with a key", func(t *testing.T) {
		t.Parallel()
		cfg := &config.Config{
			FetchBackend:     "deepseek",
			ProviderSettings: map[string]any{"deepseek_api_key": "ds-test"},
		}
		s := scraperFromConfig(t, cfg)
		if s.fetcher == nil || s.fetchBackend != "deepseek" {
			t.Fatalf("fetcher = %v, backend = %q", s.fetcher, s.fetchBackend)
		}
	})

	t.Run("unknown backend fails", func(t *testing.T) {
		t.Parallel()
		_, err := NewFromConfig(&config.Config{FetchBackend: "nope"})
		if err == nil || !strings.Contains(err.Error(), "unknown fetch backend") {
			t.Fatalf("error = %v", err)
		}
	})
}
