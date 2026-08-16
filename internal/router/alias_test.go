package router

import (
	"encoding/json"
	"net/http/httptest"
	"testing"
)

// The alias is what a human types into a harness config, so it is proven
// against the fleet's real spellings: family + version survive, packaging
// (size, quant, precision, dates, shards, deployment prefixes) does not.
func TestModelAlias(t *testing.T) {
	cases := map[string]string{
		"gemma-4-26B-A4B-it-qat-UD-Q4_K_XL":                "gemma4",
		"granite-4.1-8b-Q4_K_M":                            "granite4.1",
		"llm-6000pro-deepseek-284B-q8":                     "deepseek",
		"Qwen3.6-27B-FP8":                                  "qwen3.6",
		"Qwen3.6-27B-int4-AutoRound":                       "qwen3.6",
		"DeepSeek-V4-Flash-0731-UD-Q4_K_XL":                "deepseekv4flash",
		"Llama-4-Scout-17B-16E-Instruct":                   "llama4scout",
		"gemma-4-26B-A4B-it-qat-UD-Q4_K_XL-00001-of-00005": "gemma4",
		"": "",
	}
	for name, want := range cases {
		if got := modelAlias(name); got != want {
			t.Errorf("modelAlias(%q) = %q, want %q", name, got, want)
		}
	}
}

// An alias that reduces to an auto sentinel would swallow the auto route, so
// backendAlias must refuse it.
func TestBackendAliasNeverShadowsAuto(t *testing.T) {
	b := &Backend{BackendRegistration: BackendRegistration{ID: "w1", Model: "default-Q4"}}
	if a := backendAlias(b); a != "" {
		t.Fatalf("alias %q shadows an auto sentinel", a)
	}
}

// The third spelling: a harness may name the alias from the /v1/models menu,
// case-insensitively, alongside the exact model id and worker id.
func TestBackendServesModelAlias(t *testing.T) {
	b := &Backend{BackendRegistration: BackendRegistration{ID: "llm-arcb60", Model: "/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"}}
	for _, name := range []string{"/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf", "llm-arcb60", "gemma4", "Gemma4"} {
		if !backendServesModel(b, name) {
			t.Errorf("backend must answer to %q", name)
		}
	}
	if backendServesModel(b, "granite4.1") {
		t.Error("backend must not answer to another family's alias")
	}
}

// The menu publishes the alias as the id (raw id preserved under "root"), and
// "default" advertises the union of the fleet's features — a tools request to
// default is routed to a tools-capable worker, so default really serves it.
func TestHandleModelsAliasesAndDefaultFeatures(t *testing.T) {
	reg := newTestRegistry()
	reg.upsert(BackendRegistration{
		ID: "llm-arcb60", URL: "http://a", Model: "/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
		Features: []string{"chat", "json", "tools"}, MaxConcurrency: 1, TTLSeconds: 3600,
	})
	reg.upsert(BackendRegistration{
		ID: "llm-a750", URL: "http://b", Model: "/models/granite-4.1-8b-Q4_K_M.gguf",
		Features: []string{"chat"}, MaxConcurrency: 1, TTLSeconds: 3600,
	})
	r := &Router{cfg: &Config{}, registry: reg}

	rec := httptest.NewRecorder()
	r.handleModels(rec, httptest.NewRequest("GET", "/v1/models", nil))
	var body struct {
		Data []struct {
			ID       string   `json:"id"`
			Root     string   `json:"root"`
			Features []string `json:"features"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}

	byID := map[string][]string{}
	roots := map[string]string{}
	for _, m := range body.Data {
		byID[m.ID] = m.Features
		roots[m.ID] = m.Root
	}
	if _, ok := byID["gemma4"]; !ok {
		t.Fatalf("menu must list the alias id, got %v", byID)
	}
	if roots["gemma4"] != "/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf" {
		t.Fatalf("alias row must carry the raw id under root, got %q", roots["gemma4"])
	}
	if _, ok := byID["granite4.1"]; !ok {
		t.Fatalf("menu must list the alias id, got %v", byID)
	}
	def := map[string]bool{}
	for _, f := range byID["default"] {
		def[f] = true
	}
	if !def["chat"] || !def["json"] || !def["tools"] {
		t.Fatalf("default must advertise the fleet union, got %v", byID["default"])
	}
}

// Two distinct models reducing to the same alias would make the menu row
// ambiguous, so both keep their raw ids.
func TestHandleModelsAliasCollision(t *testing.T) {
	reg := newTestRegistry()
	reg.upsert(BackendRegistration{
		ID: "w1", URL: "http://a", Model: "gemma-4-26B-UD-Q4_K_XL.gguf",
		Features: []string{"chat"}, MaxConcurrency: 1, TTLSeconds: 3600,
	})
	reg.upsert(BackendRegistration{
		ID: "w2", URL: "http://b", Model: "gemma-4-9B-Q8_0.gguf",
		Features: []string{"chat"}, MaxConcurrency: 1, TTLSeconds: 3600,
	})
	r := &Router{cfg: &Config{}, registry: reg}

	rec := httptest.NewRecorder()
	r.handleModels(rec, httptest.NewRequest("GET", "/v1/models", nil))
	var body struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	for _, m := range body.Data {
		if m.ID == "gemma4" {
			t.Fatal("a collided alias must not be published as a menu id")
		}
	}
}
