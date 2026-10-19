package search

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/1broseidon/ketch/health"
	"github.com/1broseidon/ketch/httpx"
	config "github.com/1broseidon/ketch/internal/configbase"
)

// deepseekEndpoint is the hosted DeepSeek Search and Fetch APIs. Auth is
// Bearer-only in the Authorization header — never a query param or body field —
// so transport errors cannot leak the key via a URL.
// See https://api-docs.deepseek.com/guides/search/.
const (
	deepseekSearchEndpoint = "https://api.deepseek.com/v1/tools/search"
	deepseekFetchEndpoint  = "https://api.deepseek.com/v1/tools/fetch"

	// deepseekMaxResults is the documented upper bound of the `count` field.
	// A larger --limit is clamped rather than rejected: the operator asked for
	// as many results as the provider can return.
	deepseekMaxResults = 50
)

// DeepSeek searches the web through DeepSeek's in-house search API and, when
// selected as the fetch backend, retrieves single pages as markdown through
// the matching Fetch API.
type DeepSeek struct {
	keys   keyPool
	client *http.Client
}

// NewDeepSeek creates a new DeepSeek search backend.
func NewDeepSeek(apiKey string) *DeepSeek {
	return newDeepSeekWithKeys([]string{apiKey})
}

func newDeepSeekWithKeys(keys []string) *DeepSeek {
	return &DeepSeek{keys: newKeyPool(keys), client: httpx.Default()}
}

type deepseekSearchRequest struct {
	Query       string `json:"query"`
	Count       int    `json:"count"`
	NeedContent bool   `json:"need_content"`
}

type deepseekSearchResponse struct {
	Results []deepseekResult `json:"results"`
}

type deepseekResult struct {
	RerankScore float64 `json:"rerank_score"`
	Title       string  `json:"title"`
	URL         string  `json:"url"`
	Snippet     string  `json:"snippet"`
	Summary     string  `json:"summary"`
	Content     string  `json:"content"`
	SiteName    *string `json:"site_name"`
	SiteIcon    *string `json:"site_icon"`
	Date        *string `json:"date"`
	Authority   int     `json:"authority"`
}

type deepseekFetchRequest struct {
	URL string `json:"url"`
}

type deepseekFetchResponse struct {
	Title   string `json:"title"`
	Content string `json:"content"`
}

// Search queries DeepSeek and returns up to limit results. The page body is
// not requested here (need_content stays false): plain search should stay
// small, and full page content belongs to the scrape surface, where the fetch
// backend below can supply it.
func (d *DeepSeek) Search(ctx context.Context, query string, limit int) ([]Result, error) {
	if limit <= 0 {
		return []Result{}, nil
	}

	body, err := json.Marshal(deepseekSearchRequest{
		Query: query,
		Count: min(limit, deepseekMaxResults),
	})
	if err != nil {
		return nil, err
	}

	resp, key, err := d.post(ctx, deepseekSearchEndpoint, body)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, deepseekResponseError("search", resp, d.keys.keyLabel(key))
	}

	var sr deepseekSearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&sr); err != nil {
		return nil, fmt.Errorf("failed to decode deepseek search response: %w", err)
	}
	return mapDeepseekResults(sr.Results, limit), nil
}

// FetchPage retrieves one page as markdown through the DeepSeek Fetch API.
// It implements PageFetcher, so `fetch_backend deepseek` can serve the scrape
// surface without the built-in HTTP pipeline.
func (d *DeepSeek) FetchPage(ctx context.Context, rawURL string) (title, markdown string, err error) {
	body, err := json.Marshal(deepseekFetchRequest{URL: rawURL})
	if err != nil {
		return "", "", err
	}

	resp, key, err := d.post(ctx, deepseekFetchEndpoint, body)
	if err != nil {
		return "", "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", "", deepseekResponseError("fetch", resp, d.keys.keyLabel(key))
	}

	var fr deepseekFetchResponse
	if err := json.NewDecoder(resp.Body).Decode(&fr); err != nil {
		return "", "", fmt.Errorf("failed to decode deepseek fetch response: %w", err)
	}
	if strings.TrimSpace(fr.Content) == "" {
		return "", "", fmt.Errorf("deepseek: fetch returned no content for %s", rawURL)
	}
	return fr.Title, fr.Content, nil
}

// post sends a JSON request, rotating to another key when the first response
// is retryable and the pool has more than one. Both endpoints share the same
// auth and rate-limit behaviour, so they share the retry policy.
func (d *DeepSeek) post(ctx context.Context, endpoint string, body []byte) (*http.Response, string, error) {
	key := d.keys.pick()
	resp, err := d.request(ctx, endpoint, body, key)
	if err != nil {
		return nil, "", err
	}
	if deepseekRetryableStatus(resp.StatusCode) && d.keys.size() > 1 {
		closeSearchResponse(resp)
		key = d.keys.pickDifferent(key)
		resp, err = d.request(ctx, endpoint, body, key)
		if err != nil {
			return nil, "", err
		}
	}
	return resp, key, nil
}

func (d *DeepSeek) request(ctx context.Context, endpoint string, body []byte, key string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	resp, err := d.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("deepseek request failed: %w", err)
	}
	return resp, nil
}

// deepseekRetryableStatus lists the outcomes another key might survive: bad
// credentials and rate limits are per key, and 5xx is a provider-side failure
// that costs nothing to retry (failed requests are not billed).
func deepseekRetryableStatus(status int) bool {
	return status == http.StatusUnauthorized ||
		status == http.StatusTooManyRequests ||
		status >= http.StatusInternalServerError
}

// deepseekResponseError maps the documented statuses to actionable errors.
// op is "search" or "fetch"; the Fetch API has two statuses the Search API
// does not (403 policy rejection, 404 unretrievable page) with their own
// advice.
func deepseekResponseError(op string, resp *http.Response, keyLabel string) error {
	detail := deepseekErrorDetail(resp)
	switch resp.StatusCode {
	case http.StatusBadRequest:
		return fmt.Errorf("deepseek %s: invalid request: %s", op, deepseekDetailOr(detail, "the request was rejected"))
	case http.StatusUnauthorized:
		return fmt.Errorf("deepseek: invalid API key (%s; set via: ketch config set deepseek_api_key <key>)", keyLabel)
	case http.StatusPaymentRequired:
		return fmt.Errorf("deepseek: account balance exhausted — top up at https://platform.deepseek.com (%s)", keyLabel)
	case http.StatusForbidden:
		return fmt.Errorf("deepseek %s: rejected by the security policy: %s", op, deepseekDetailOr(detail, "no detail"))
	case http.StatusNotFound:
		return fmt.Errorf("deepseek %s: no usable page content: %s", op, deepseekDetailOr(detail, "the URL is unavailable, blocked, or has no extractable content"))
	case http.StatusRequestTimeout, http.StatusGatewayTimeout:
		return fmt.Errorf("deepseek %s: upstream timed out (%s)", op, keyLabel)
	case http.StatusRequestEntityTooLarge:
		return fmt.Errorf("deepseek %s: request body too large", op)
	case http.StatusTooManyRequests:
		return fmt.Errorf("deepseek: rate limited (%s)", keyLabel)
	default:
		if detail != "" {
			return fmt.Errorf("deepseek %s returned status %d: %s", op, resp.StatusCode, detail)
		}
		return fmt.Errorf("deepseek %s returned status %d", op, resp.StatusCode)
	}
}

// deepseekDetailOr replaces an empty upstream detail with a fallback so the
// message never ends in a dangling colon.
func deepseekDetailOr(detail, fallback string) string {
	if detail != "" {
		return detail
	}
	return fallback
}

// deepseekErrorDetail reads a bounded body and returns the OpenAI-compatible
// error message when the provider sent one, otherwise the raw body text.
func deepseekErrorDetail(resp *http.Response) string {
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	detail := strings.TrimSpace(string(raw))
	if detail == "" {
		return ""
	}
	var envelope struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &envelope); err == nil && envelope.Error.Message != "" {
		return strings.TrimSpace(envelope.Error.Message)
	}
	return detail
}

func mapDeepseekResults(raw []deepseekResult, limit int) []Result {
	results := make([]Result, 0, limit)
	for _, r := range raw {
		if len(results) >= limit {
			break
		}
		if strings.TrimSpace(r.URL) == "" {
			continue
		}
		description := r.Snippet
		if strings.TrimSpace(description) == "" {
			description = r.Summary
		}
		results = append(results, Result{
			Title:       r.Title,
			URL:         r.URL,
			Description: description,
			Content:     r.Content,
		})
	}
	return results
}

const deepseekProbeBody = `{"query":"ketch","count":1}`

// ProbeDeepSeek checks the provider using a caller-supplied client.
func ProbeDeepSeek(ctx context.Context, client *http.Client, apiKey string) (health.Status, string) {
	key := strings.TrimSpace(apiKey)
	if key == "" {
		return health.StatusNoKey, "API key not set (get one at https://platform.deepseek.com then: ketch config set deepseek_api_key <key>)"
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, deepseekSearchEndpoint, strings.NewReader(deepseekProbeBody))
	if err != nil {
		return health.StatusUnreachable, health.ErrorDetail(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+key)

	resp, err := client.Do(req)
	if err != nil {
		return health.StatusUnreachable, health.ErrorDetail(err)
	}
	defer health.Drain(resp)

	switch resp.StatusCode {
	case http.StatusOK:
		return health.StatusOK, ""
	case http.StatusUnauthorized, http.StatusForbidden:
		return health.StatusMisconfigured, "API key rejected (ketch config set deepseek_api_key <key>)"
	case http.StatusPaymentRequired:
		return health.StatusOK, "reachable, key accepted (account balance exhausted)"
	case http.StatusTooManyRequests:
		return health.StatusOK, "reachable, key accepted (rate limited)"
	default:
		return health.StatusUnreachable, fmt.Sprintf("returned status %d", resp.StatusCode)
	}
}

func deepseekProvider() Provider {
	return Provider{
		// The in-house search engine is slower than the hosted SERP APIs;
		// allow a doctor probe to wait longer than the default budget.
		MinProbeTimeout: 10 * time.Second,
		Settings:        []config.Setting{config.KeyPool("deepseek_api_key", "deepseek_api_keys")},
		AutoRank:        35,
		ID:              "deepseek",
		Setup:           "deepseek: API key not set (get one at https://platform.deepseek.com then: ketch config set deepseek_api_key <key>)",
		Name:            "DeepSeek",
		Fetches:         true,
		Usable:          func(c *config.Config) bool { return len(c.DeepSeekKeys()) > 0 },
		New:             func(c *config.Config) (Searcher, error) { return newDeepSeekWithKeys(c.DeepSeekKeys()), nil },
		Probe: func(ctx context.Context, client *http.Client, c *config.Config) (health.Status, string) {
			return health.ProbeKeyPool(c.DeepSeekKeys(), func(key string) (health.Status, string) {
				return ProbeDeepSeek(ctx, client, key)
			})
		},
	}
}
