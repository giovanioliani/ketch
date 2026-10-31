package search

import "context"

// Result represents a single search result.
type Result struct {
	Title       string `json:"title"`
	URL         string `json:"url"`
	FetchedURL  string `json:"fetched_url,omitempty"`
	Description string `json:"description,omitempty"`
	Content     string `json:"content,omitempty"`
	// Backends lists the backends that returned this result under federated
	// (--multi) search. Empty (and omitted) for single-backend results, so
	// every existing single-backend output stays byte-identical.
	Backends []string `json:"backends,omitempty"`
}

// Searcher is the interface for search backends.
type Searcher interface {
	Search(ctx context.Context, query string, limit int) ([]Result, error)
}

// PageFetcher is implemented by providers that can retrieve a single page as
// markdown through their own API. Selecting such a provider as the fetch
// backend lets `ketch scrape` and `search --scrape` use that API instead of
// the built-in HTTP fetch + extraction pipeline (which still handles --raw,
// --select, --force-browser, and crawl, where raw HTML is required).
type PageFetcher interface {
	FetchPage(ctx context.Context, rawURL string) (title, markdown string, err error)
}
