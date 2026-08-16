package router

// Price seeding from LiteLLM's public price table.
//
// An operator entering a metered endpoint knows the model id and the base URL.
// They should not also have to look up two prices and a context window, retype
// them, and get a decimal place wrong — the numbers are public, they are
// maintained, and LiteLLM already keeps them in one MIT-licensed file that the
// whole ecosystem reads.
//
// So the snapshot is EMBEDDED rather than fetched at runtime, for the same
// reason benchdata/pool.json is (see benchgen.go): a router that phones home on
// startup has an egress dependency on a box that may not have one, and a price
// table that can change underneath a running deployment makes a stored cost
// impossible to reconcile against the run that produced it. `discrimen prices
// fetch` refreshes it as a reviewed commit.
//
// Seeding is BEST-EFFORT and never load-bearing. An empty snapshot, a model id
// nobody publishes a price for, a provider that spells it differently — all of
// them mean the operator types the number themselves, which is what they would
// have done anyway. Nothing downstream may assume a price was found.

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"
)

// priceData is the trimmed LiteLLM snapshot. Refreshed by `discrimen prices
// fetch`; priceSourceURL is where it comes from and
// pricedata/LICENSE-litellm.txt carries its MIT notice.
//
// NOTE: go:embed requires the file to EXIST at compile time, so an empty
// snapshot is committed as `{}` rather than deleted. Everything below tolerates
// that, and the tests run against it.
//
//go:embed pricedata/prices.json
var priceData []byte

// priceEntry is the four fields of a LiteLLM row this router uses. Costs are
// per single token, as LiteLLM stores them; the registry holds price per
// million, because that is the unit every provider quotes and the one an
// operator can sanity-check at a glance.
type priceEntry struct {
	InputCostPerToken  float64 `json:"input_cost_per_token,omitempty"`
	OutputCostPerToken float64 `json:"output_cost_per_token,omitempty"`
	// MaxInputTokens is LiteLLM's max_input_tokens, falling back to its
	// max_tokens where a row publishes only that.
	MaxInputTokens int    `json:"max_input_tokens,omitempty"`
	Provider       string `json:"litellm_provider,omitempty"`
}

type priceSnapshot struct {
	Source    string                `json:"source"`
	FetchedAt string                `json:"fetched_at"`
	Models    map[string]priceEntry `json:"models"`
}

// priceTable is the parsed snapshot plus the tolerant lookup index over it.
//
// The two maps are the whole trick. exact holds only the snapshot's own keys, so
// a query that names a model exactly as LiteLLM does gets LiteLLM's answer.
// reduced holds the lossy spellings — basenames, date-stripped ids — where
// collisions are unavoidable: "azure/gpt-4o" and "gpt-4o" both reduce to
// "gpt-4o", and their prices differ. One flat map would let the azure row's
// basename shadow OpenAI's exact key, and quietly bill the wrong number.
type priceTable struct {
	snapshot priceSnapshot
	exact    map[string]priceEntry
	reduced  map[string]priceEntry
}

var (
	priceOnce   sync.Once
	priceLoaded *priceTable
)

// prices parses and indexes the embedded snapshot on first use.
func prices() *priceTable {
	priceOnce.Do(func() { priceLoaded = newPriceTable(priceData) })
	return priceLoaded
}

// newPriceTable parses a snapshot and builds the lookup index. A snapshot that
// is empty, `{}`, or unparseable yields an EMPTY table rather than an error:
// seeding is an assist, and a broken assist must not stop a router starting or
// refuse to create a provider row.
func newPriceTable(data []byte) *priceTable {
	t := &priceTable{exact: map[string]priceEntry{}, reduced: map[string]priceEntry{}}
	if len(bytes.TrimSpace(data)) == 0 {
		return t
	}
	if err := json.Unmarshal(data, &t.snapshot); err != nil {
		return t
	}
	keys := make([]string, 0, len(t.snapshot.Models))
	for key := range t.snapshot.Models {
		keys = append(keys, key)
	}
	// Unprefixed keys first, then lexical. Two things ride on this order, and
	// both want the same thing: collisions in the reduced index have to resolve
	// identically on every build (not in Go's randomised map order), and where
	// several rows reduce to one spelling the least-qualified is the closest
	// thing to a canonical answer — "gpt-4o" is OpenAI's own price, where
	// "azure/gpt-4o" is one reseller's.
	sort.Slice(keys, func(i, j int) bool {
		si, sj := strings.Count(keys[i], "/"), strings.Count(keys[j], "/")
		if si != sj {
			return si < sj
		}
		return keys[i] < keys[j]
	})
	for _, key := range keys {
		entry := t.snapshot.Models[key]
		lower := strings.ToLower(strings.TrimSpace(key))
		if _, taken := t.exact[lower]; !taken {
			t.exact[lower] = entry
		}
		for _, spelling := range priceReducedSpellings(lower) {
			if _, taken := t.reduced[spelling]; !taken {
				t.reduced[spelling] = entry
			}
		}
	}
	return t
}

// priceDateSuffix matches the release stamp providers append to an otherwise
// stable model id — claude-sonnet-4-5-20250929, gpt-4o-2024-08-06 — so the two
// spellings of the same weights resolve to the same row.
var priceDateSuffix = regexp.MustCompile(`-(\d{8}|\d{4}-\d{2}-\d{2})$`)

// priceRevisionSuffix matches Vertex's @001 revision marker.
var priceRevisionSuffix = regexp.MustCompile(`@\d+$`)

// reducePriceKey strips the decorations that distinguish spellings of the same
// model rather than different models: case, a release date, a Vertex revision,
// and an OpenRouter variant suffix (":free", ":extended").
//
// It is applied to BOTH the snapshot keys and the lookup, so the two meet in the
// middle. It is deliberately not applied to the provider prefix — that is
// handled separately, because dropping it is a much bigger reduction and wants
// to be the last thing tried.
func reducePriceKey(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	// Bedrock ids carry a genuine ":0" version, so cut at the LAST colon only
	// when what follows is not a bare number.
	if i := strings.LastIndex(s, ":"); i > 0 && !isAllDigits(s[i+1:]) {
		s = s[:i]
	}
	s = priceRevisionSuffix.ReplaceAllString(s, "")
	s = priceDateSuffix.ReplaceAllString(s, "")
	return s
}

// priceReducedSpellings enumerates the LOSSY forms of a model id, narrowest
// first. The basename forms are what let a query for "grok-4" find the row
// LiteLLM stores as "xai/grok-4", and vice versa. Used both to index the
// snapshot and to widen a lookup, so the two meet in the middle.
func priceReducedSpellings(lower string) []string {
	out := []string{reducePriceKey(lower)}
	if i := strings.LastIndex(lower, "/"); i >= 0 && i+1 < len(lower) {
		base := lower[i+1:]
		out = append(out, base, reducePriceKey(base))
	}
	return out
}

// lookupPrice finds the snapshot row for a model id, tolerantly. provider is the
// row's declared provider and is only a hint: LiteLLM stores the same model
// under a bare id ("gpt-4o"), under "<vendor>/<id>" ("xai/grok-4") and under
// several gateways' prefixes, and which one a given model landed under is not
// something an operator should have to know.
//
// Order matters and is the reason this is not one map lookup. Every EXACT
// spelling is tried before any reduced one, with the provider-qualified form
// first — "mistral/mistral-large-latest" is priced and "mistral-large-latest" is
// not, while "gpt-4o" is priced by OpenAI and "azure/gpt-4o" differently by
// Azure. Only once nothing matches exactly does the basename fallback run.
//
// ok=false means nothing matched, which is a completely normal outcome.
func lookupPrice(model, provider string) (priceEntry, bool) {
	return prices().lookup(model, provider)
}

func (t *priceTable) lookup(model, provider string) (priceEntry, bool) {
	model = strings.ToLower(strings.TrimSpace(model))
	if model == "" || len(t.exact) == 0 {
		return priceEntry{}, false
	}
	provider = strings.ToLower(strings.TrimSpace(provider))
	candidates := []string{}
	if provider != "" && provider != providerLocal {
		candidates = append(candidates, provider+"/"+model)
	}
	candidates = append(candidates, model)
	for _, c := range candidates {
		if e, ok := t.exact[c]; ok {
			return e, true
		}
	}
	for _, c := range candidates {
		for _, spelling := range priceReducedSpellings(c) {
			if e, ok := t.reduced[spelling]; ok {
				return e, true
			}
		}
	}
	return priceEntry{}, false
}

// priceStated marks the fields a caller named EXPLICITLY, which seeding must
// then leave alone.
//
// It exists because "the operator left this blank" and "the operator said zero"
// are different statements that arrive at seedPrices looking identical. An
// explicit 0 declares a model free — a real thing on an otherwise metered
// endpoint — and without this it would be unsayable: every write would re-seed
// the published price straight back over it.
type priceStated struct {
	Input   bool
	Output  bool
	Context bool
}

// seedPrices fills the blanks on a registration from the embedded snapshot and
// reports which fields it touched, so the admin API can say what it did rather
// than silently inventing numbers on an operator's row.
//
// It only ever fills a field that is both zero and unstated. That is the same
// rule the rest of P2 runs on: refine what was left blank, never overwrite what
// was entered.
func seedPrices(reg *BackendRegistration, stated priceStated) []string {
	return seedPricesWith(prices(), reg, stated)
}

func seedPricesWith(t *priceTable, reg *BackendRegistration, stated priceStated) []string {
	entry, ok := t.lookup(reg.Model, reg.Provider)
	if !ok {
		return nil
	}
	var filled []string
	if !stated.Input && reg.InputPricePerMtok == 0 && entry.InputCostPerToken > 0 {
		reg.InputPricePerMtok = entry.InputCostPerToken * 1e6
		filled = append(filled, "input_price_per_mtok")
	}
	if !stated.Output && reg.OutputPricePerMtok == 0 && entry.OutputCostPerToken > 0 {
		reg.OutputPricePerMtok = entry.OutputCostPerToken * 1e6
		filled = append(filled, "output_price_per_mtok")
	}
	// Context is a seed only. queryContextMeasured re-reads it from the endpoint
	// on every certification and, on a manual row, only when the operator left it
	// blank — so a published window that turns out to be wrong is corrected by
	// measurement, which is the order this router does everything else in.
	if !stated.Context && reg.ContextK == 0 && entry.MaxInputTokens >= 1024 {
		reg.ContextK = entry.MaxInputTokens / 1024
		filled = append(filled, "context_k")
	}
	return filled
}

func isAllDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}

// ── Dev subcommand: `discrimen prices fetch` ────────────────────────────────
//
// Same shape and the same reasoning as `discrimen bench fetch` (benchgen.go):
// it touches the network, it writes into the source tree, and it is run from a
// checkout by a person who then reviews the diff. It is never invoked in the
// container.

// priceSourceURL is LiteLLM's price table. MIT licensed; the notice lives in
// pricedata/LICENSE-litellm.txt next to the snapshot.
const priceSourceURL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

func priceDataDir() string {
	_, self, _, ok := runtime.Caller(0)
	if !ok {
		return "pricedata"
	}
	return filepath.Join(filepath.Dir(self), "pricedata")
}

func pricesUsage() {
	fmt.Fprint(os.Stderr, `discrimen prices — refresh the embedded LiteLLM price snapshot

  prices fetch     Pull `+priceSourceURL+`, trim it, and rewrite
                   internal/router/pricedata/prices.json

Trimmed to the four fields the router reads (input and output cost per token,
context window, provider) and to rows that publish a per-token price at all —
the full table is ~1.7MB of image, audio and embedding rows this never looks at.

  go run . prices fetch
  # then review the diff and commit it
`)
}

// runPricesCommand dispatches `discrimen prices …`. Returns false if argv isn't
// a prices invocation, so Main can carry on and start the server.
func runPricesCommand(argv []string) bool {
	if len(argv) < 2 || argv[1] != "prices" {
		return false
	}
	if len(argv) < 3 || argv[2] != "fetch" {
		pricesUsage()
		os.Exit(2)
	}
	fs := flag.NewFlagSet("fetch", flag.ExitOnError)
	_ = fs.Parse(argv[3:])
	if err := pricesFetch(); err != nil {
		fmt.Fprintf(os.Stderr, "prices fetch: %v\n", err)
		os.Exit(1)
	}
	return true
}

func pricesFetch() error {
	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Get(priceSourceURL)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: HTTP %d", priceSourceURL, resp.StatusCode)
	}
	var raw map[string]map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
		return err
	}

	models := map[string]priceEntry{}
	for key, row := range raw {
		// LiteLLM ships a self-documenting "sample_spec" row whose values are
		// field descriptions rather than numbers.
		if key == "sample_spec" {
			continue
		}
		e := priceEntry{
			InputCostPerToken:  jsonNum(row, "input_cost_per_token"),
			OutputCostPerToken: jsonNum(row, "output_cost_per_token"),
		}
		// Rows with no per-token price are image, audio, embedding and rerank
		// models. Keeping them would quadruple the file for data this never reads.
		if e.InputCostPerToken <= 0 && e.OutputCostPerToken <= 0 {
			continue
		}
		if v := jsonNum(row, "max_input_tokens"); v > 0 {
			e.MaxInputTokens = int(v)
		} else if v := jsonNum(row, "max_tokens"); v > 0 {
			e.MaxInputTokens = int(v)
		}
		e.Provider, _ = row["litellm_provider"].(string)
		models[key] = e
	}
	if len(models) == 0 {
		return fmt.Errorf("no priced models in the upstream table — has its shape changed?")
	}

	path := filepath.Join(priceDataDir(), "prices.json")
	if err := os.MkdirAll(priceDataDir(), 0o755); err != nil {
		return err
	}
	buf, err := renderPriceSnapshot(priceSnapshot{
		Source:    priceSourceURL,
		FetchedAt: time.Now().UTC().Format(time.RFC3339),
		Models:    models,
	})
	if err != nil {
		return err
	}
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "wrote %s (%d models, %d bytes)\n", path, len(models), len(buf))
	return nil
}

// renderPriceSnapshot writes the snapshot with one model per line, keys sorted.
// json.MarshalIndent would put every field of every row on its own line and add
// 65KB of whitespace to a file that is already at the size where it stops being
// reviewable; one compact line per model keeps the diff of a refresh readable —
// a changed price is one changed line.
func renderPriceSnapshot(snap priceSnapshot) ([]byte, error) {
	keys := make([]string, 0, len(snap.Models))
	for k := range snap.Models {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var out bytes.Buffer
	out.WriteString("{\n")
	src, _ := json.Marshal(snap.Source)
	at, _ := json.Marshal(snap.FetchedAt)
	fmt.Fprintf(&out, " \"source\": %s,\n \"fetched_at\": %s,\n \"models\": {\n", src, at)
	for i, k := range keys {
		name, err := json.Marshal(k)
		if err != nil {
			return nil, err
		}
		entry, err := json.Marshal(snap.Models[k])
		if err != nil {
			return nil, err
		}
		sep := ","
		if i == len(keys)-1 {
			sep = ""
		}
		fmt.Fprintf(&out, "  %s: %s%s\n", name, entry, sep)
	}
	out.WriteString(" }\n}\n")
	return out.Bytes(), nil
}
