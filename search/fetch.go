package search

import (
	"fmt"
	"strings"

	config "github.com/1broseidon/ketch/internal/configbase"
)

// FetchBackendHTTP is the built-in scrape pipeline: fetch the page over HTTP,
// extract readable markdown locally, and fall back to a browser for JS shells.
// It is the default and the only value that needs no credentials.
const FetchBackendHTTP = "http"

// FetchBackends returns every accepted fetch_backend value: the built-in HTTP
// pipeline first, then providers whose API can return page markdown.
func FetchBackends() []string {
	names := []string{FetchBackendHTTP}
	for _, p := range providers {
		if !p.Hidden && p.Fetches {
			names = append(names, p.ID)
		}
	}
	return names
}

// IsFetchBackend reports whether id names a selectable fetch backend.
func IsFetchBackend(id string) bool {
	backend := effectiveFetchBackendName(id)
	if backend == FetchBackendHTTP {
		return true
	}
	p, ok := Lookup(backend)
	return ok && p.Fetches
}

// EffectiveFetchBackend returns the fetch backend ketch will use for cfg,
// defaulting to the built-in HTTP pipeline when unset.
func EffectiveFetchBackend(cfg *config.Config) string {
	return effectiveFetchBackendName(cfg.FetchBackend)
}

// effectiveFetchBackendName lowercases a raw value so config set, env
// overrides, and hand-edited files all resolve the same way.
func effectiveFetchBackendName(value string) string {
	backend := strings.ToLower(strings.TrimSpace(value))
	if backend == "" {
		return FetchBackendHTTP
	}
	return backend
}

// NewFetcherFromConfig resolves the operator's fetch backend into a
// PageFetcher. It returns a nil fetcher for the built-in HTTP pipeline, which
// the scrape package implements itself. Construction performs no network I/O;
// missing credentials surface here as a precondition error.
func NewFetcherFromConfig(cfg *config.Config) (PageFetcher, error) {
	backend := EffectiveFetchBackend(cfg)
	if backend == FetchBackendHTTP {
		return nil, nil
	}
	p, ok := Lookup(backend)
	if !ok || !p.Fetches {
		return nil, fmt.Errorf("unknown fetch backend %q (available: %s)", backend, strings.Join(FetchBackends(), ", "))
	}
	searcher, err := p.Build(cfg)
	if err != nil {
		return nil, fmt.Errorf("fetch backend %s: %w", backend, err)
	}
	fetcher, ok := searcher.(PageFetcher)
	if !ok {
		return nil, fmt.Errorf("fetch backend %s cannot fetch pages", backend)
	}
	return fetcher, nil
}
